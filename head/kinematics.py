"""
head/kinematics.py
==================
XLeRobot 2-DOF camera head kinematics library.

FK/IK functions and math utilities.  Import from here; do not run directly.

    from kinematics import fk, fk_axis, fk_position, ik, ik_look_at

Conventions
-----------
- Motor values are normalised to [-100, 100].
- motor1 (ID 1) → tilt / pitch  (rotates around Y)
- motor2 (ID 2) → pan  / yaw   (rotates around Z)
- All positions in metres, in the top_base_link frame.

URDF joint origins
------------------
  head_pan_joint  : xyz="-0.103 0 0.323"      rpy="0 0 0"   axis Z
  head_tilt_joint : xyz=" 0.001 0.002 0.09815" rpy="0 0 0"   axis Y
  camera_fixed    : xyz=" 0.025 0 0.03"        rpy="0 0 0"
"""

from __future__ import annotations
import numpy as np


# ── Math utilities ─────────────────────────────────────────────────────────

def motor_to_rad(val: float, motor_id: int = 1) -> float:
    """Normalised motor value [-100, 100] → radians [-π/2, π/2]. motor2 is sign-flipped."""
    sign = -1 if motor_id == 2 else 1
    return sign * val / 100.0 * (np.pi / 2)


def rad_to_motor(rad: float, motor_id: int = 1) -> float:
    """Radians → normalised motor value. motor2 is sign-flipped."""
    sign = -1 if motor_id == 2 else 1
    return sign * rad / (np.pi / 2) * 100.0


def htm(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """3×3 rotation + 3-vector translation → 4×4 homogeneous transform."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]])


# ── URDF constant transforms ───────────────────────────────────────────────

T_PAN_ORIGIN  = htm(np.eye(3), np.array([-0.103, 0.000,   0.323  ]))
T_TILT_ORIGIN = htm(np.eye(3), np.array([ 0.001, 0.002,   0.09815]))
T_CAM_FIXED   = htm(np.eye(3), np.array([ 0.025, 0.000,   0.030  ]))


# ── Forward kinematics ─────────────────────────────────────────────────────

def _intermediate(motor1: float, motor2: float):
    """Return (T_base, T_pan, T_tilt, T_cam) — the full kinematic chain."""
    T_pan  = htm(rot_z(motor_to_rad(motor2, 2)), np.zeros(3))
    T_tilt = htm(rot_y(motor_to_rad(motor1, 1)), np.zeros(3))

    T_base      = np.eye(4)
    T_pan_node  = T_PAN_ORIGIN  @ T_pan
    T_tilt_node = T_pan_node    @ T_TILT_ORIGIN @ T_tilt
    T_cam       = T_tilt_node   @ T_CAM_FIXED
    return T_base, T_pan_node, T_tilt_node, T_cam


def fk(motor1: float, motor2: float) -> np.ndarray:
    """Full camera pose in top_base_link frame as a 4×4 HTM."""
    return _intermediate(motor1, motor2)[3]


def fk_position(motor1: float, motor2: float) -> np.ndarray:
    """Camera origin (x, y, z) in top_base_link, metres."""
    return fk(motor1, motor2)[:3, 3]


def fk_axis(motor1: float, motor2: float) -> np.ndarray:
    """Camera optical axis (X-axis) unit vector in top_base_link."""
    return fk(motor1, motor2)[:3, 0]


# ── Inverse kinematics ─────────────────────────────────────────────────────
#
# Optical-axis derivation (motor2 is sign-flipped → θ_pan_actual = -motor2·π/200):
#   axis = Rz(θ_pan) @ Ry(θ_tilt) @ [1,0,0]ᵀ
#        = [cos θ_pan · cos θ_tilt,
#           sin θ_pan · cos θ_tilt,
#          -sin θ_tilt             ]
# Solution:
#   θ_tilt = arcsin(-dz),  θ_pan = atan2(dy, dx)

def ik(direction: np.ndarray, clip: bool = True) -> tuple[float, float]:
    """
    Direction vector (top_base_link frame, need not be unit) → (motor1, motor2).
    Returns normalised motor values [-100, 100].
    """
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    dx, dy, dz = d

    theta_tilt = np.arcsin(-dz)
    theta_pan  = np.arctan2(dy, dx)

    m1 = rad_to_motor(theta_tilt, 1)
    m2 = rad_to_motor(theta_pan,  2)

    if clip:
        m1 = float(np.clip(m1, -100, 100))
        m2 = float(np.clip(m2, -100, 100))
    return m1, m2


def ik_look_at(target: np.ndarray,
               camera_origin: np.ndarray | None = None,
               motor1: float = 0.0,
               motor2: float = 0.0,
               clip: bool = True) -> tuple[float, float]:
    """
    Compute motor values that point the camera optical axis at *target*.
    camera_origin defaults to the FK result for the given motor values.
    """
    if camera_origin is None:
        camera_origin = fk_position(motor1, motor2)
    return ik(np.asarray(target) - np.asarray(camera_origin), clip)
