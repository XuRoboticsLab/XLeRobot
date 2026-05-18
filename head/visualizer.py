"""
head/visualizer.py
==================
3-D live kinematics visualizer and motor data sources for the camera head.

Typical usage
-------------
    from visualizer import run_visualizer, DemoSource, HardwareSource, LookAtSource

    # No hardware — sine sweep
    run_visualizer(DemoSource())

    # Real motors, free motion
    src = HardwareSource(port="/dev/ttyACM0")
    run_visualizer(src)
    src.close()

    # IK look-at a fixed point
    import numpy as np
    src = LookAtSource(np.array([0.5, 0.0, 0.4]))
    run_visualizer(src, target_point=np.array([0.5, 0.0, 0.4]))
    src.close()
"""

from __future__ import annotations
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from kinematics import _intermediate, motor_to_rad, ik_look_at


_AXIS_COLORS = ["#e74c3c", "#2ecc71", "#3498db"]   # X=red  Y=green  Z=blue
_AXIS_LABELS = ["X", "Y", "Z"]


# ── Motor data sources ─────────────────────────────────────────────────────

class DemoSource:
    """No hardware: sinusoidal sweep of both axes."""

    def __init__(self) -> None:
        self._t0 = time.time()

    def __call__(self) -> tuple[float, float]:
        t  = time.time() - self._t0
        m1 = 50 * np.sin(t * 0.7)
        m2 = 80 * np.sin(t * 0.4 + 1.0)
        return float(m1), float(m2)


class HardwareSource:
    """Live motor readback from a real head (torque disabled — free motion)."""

    def __init__(self, port: str = "/dev/ttyACM0") -> None:
        from head_motor import HeadMotorController
        self._head = HeadMotorController(port=port)
        self._head.connect()
        self._head.bus.disable_torque()
        self._last: tuple[float, float] = (0.0, 0.0)

    def __call__(self) -> tuple[float, float]:
        try:
            pos = self._head.get_pos()
            self._last = (
                pos.get("head_motor_1", 0.0),
                pos.get("head_motor_2", 0.0),
            )
        except Exception:
            pass
        return self._last

    def close(self) -> None:
        self._head.disconnect()


class LookAtSource:
    """Drive the real head to look at a fixed target; read back actual positions for visualization."""

    def __init__(self, target: np.ndarray, port: str = "/dev/ttyACM0") -> None:
        from head_motor import HeadMotorController
        self._m1, self._m2 = ik_look_at(np.asarray(target, dtype=float))
        self._head = HeadMotorController(port=port)
        self._head.connect()
        self._last: tuple[float, float] = (self._m1, self._m2)

    def __call__(self) -> tuple[float, float]:
        self._head.send_pos(self._m1, self._m2)
        try:
            pos = self._head.get_pos()
            self._last = (
                pos.get("head_motor_1", self._m1),
                pos.get("head_motor_2", self._m2),
            )
        except Exception:
            pass
        return self._last

    def close(self) -> None:
        self._head.disconnect()


# ── Visualizer ─────────────────────────────────────────────────────────────

def run_visualizer(
    motor_source,
    interval_ms: int = 50,
    target_point: np.ndarray | None = None,
) -> None:
    """
    Launch a blocking 3-D animation of the head kinematic chain.

    Parameters
    ----------
    motor_source  : callable () -> (motor1, motor2)
    interval_ms   : animation update interval in milliseconds
    target_point  : optional 3-D point (metres) shown as a red sphere
    """

    def _draw_frame(ax, T, length=0.06, label="", alpha=1.0):
        origin = T[:3, 3]
        for i, (color, lbl) in enumerate(zip(_AXIS_COLORS, _AXIS_LABELS)):
            axis_vec = T[:3, i]
            ax.quiver(*origin, *(axis_vec * length),
                      color=color, alpha=alpha,
                      linewidth=2.5, arrow_length_ratio=0.25)
            tip = origin + axis_vec * length * 1.2
            ax.text(*tip, f"{label}.{lbl}", color=color,
                    fontsize=7, alpha=alpha, ha="center", va="center")

    def _draw_link(ax, T0, T1):
        p0, p1 = T0[:3, 3], T1[:3, 3]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]],
                color="#7f8c8d", linewidth=1.5, linestyle="--", alpha=0.6)

    def _style(ax):
        if target_point is not None:
            m = 0.05
            xl = max(0.25, abs(target_point[0]) + m)
            yl = max(0.25, abs(target_point[1]) + m)
            zh = max(0.55, target_point[2] + m)
        else:
            xl, yl, zh = 0.25, 0.25, 0.55
        ax.set_xlim(-xl,  xl)
        ax.set_ylim(-yl,  yl)
        ax.set_zlim( 0.0, zh)
        ax.set_xlabel("X (m)", color="#aaaaaa", fontsize=8)
        ax.set_ylabel("Y (m)", color="#aaaaaa", fontsize=8)
        ax.set_zlabel("Z (m)", color="#aaaaaa", fontsize=8)
        ax.tick_params(colors="#555555", labelsize=7)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#333355")
        ax.grid(True, color="#2a2a4a", linewidth=0.5)
        ax.set_title("Camera Head — Live Kinematics",
                     color="white", fontsize=12, pad=10)

    fig = plt.figure(figsize=(10, 8), facecolor="#1a1a2e")
    ax  = fig.add_subplot(111, projection="3d", facecolor="#16213e")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0.10)

    info = fig.text(0.02, 0.01, "", color="white",
                    fontsize=9, fontfamily="monospace",
                    verticalalignment="bottom")

    legend_handles = (
        [plt.Line2D([0], [0], color=c, linewidth=2, label=f"{l} axis")
         for c, l in zip(_AXIS_COLORS, _AXIS_LABELS)]
        + [plt.Line2D([0], [0], color="#f39c12", linewidth=1.5,
                      linestyle=":", label="optical axis")]
    )

    def update(_frame):
        ax.cla()
        _style(ax)

        m1, m2 = motor_source()
        T_base, T_pan, T_tilt, T_cam = _intermediate(m1, m2)

        _draw_link(ax, T_base, T_pan)
        _draw_link(ax, T_pan,  T_tilt)
        _draw_link(ax, T_tilt, T_cam)

        _draw_frame(ax, T_base, length=0.055, label="BASE", alpha=0.45)
        _draw_frame(ax, T_pan,  length=0.048, label="PAN",  alpha=0.70)
        _draw_frame(ax, T_tilt, length=0.048, label="TILT", alpha=0.70)
        _draw_frame(ax, T_cam,  length=0.070, label="CAM",  alpha=1.00)

        cam_pos = T_cam[:3, 3]
        cam_xax = T_cam[:3, 0]
        tip = cam_pos + cam_xax * 0.15
        ax.plot([cam_pos[0], tip[0]],
                [cam_pos[1], tip[1]],
                [cam_pos[2], tip[2]],
                color="#f39c12", linewidth=1.5, linestyle=":", alpha=0.9)

        if target_point is not None:
            tp = target_point
            ax.scatter(*tp, s=300, c="#e74c3c", marker="o",
                       depthshade=True, zorder=6)
            ax.text(tp[0], tp[1], tp[2] + 0.012,
                    f"({tp[0]:.2f}, {tp[1]:.2f}, {tp[2]:.2f})",
                    color="#e74c3c", fontsize=7, ha="center", va="bottom")
            ax.plot([cam_pos[0], tp[0]],
                    [cam_pos[1], tp[1]],
                    [cam_pos[2], tp[2]],
                    color="#e74c3c", linewidth=1.2, linestyle="--", alpha=0.6)

        ax.legend(handles=legend_handles, loc="upper right",
                  facecolor="#1a1a2e", edgecolor="#444466",
                  labelcolor="white", fontsize=8)

        info.set_text(
            f"motor1(tilt): {m1:+7.2f}   θ_tilt = {np.degrees(motor_to_rad(m1, 1)):+6.1f}°   "
            f"motor2(pan):  {m2:+7.2f}   θ_pan  = {np.degrees(motor_to_rad(m2, 2)):+6.1f}°\n"
            f"cam origin  : [{cam_pos[0]:+.4f},  {cam_pos[1]:+.4f},  {cam_pos[2]:+.4f}] m     "
            f"cam X-axis  : [{cam_xax[0]:+.3f},  {cam_xax[1]:+.3f},  {cam_xax[2]:+.3f}]"
        )

    anim = animation.FuncAnimation(  # noqa: F841  keep reference to prevent GC
        fig, update, interval=interval_ms, cache_frame_data=False
    )
    plt.show()
