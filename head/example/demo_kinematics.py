"""
FK/IK math verification + 3-D live visualizer CLI.

Usage
-----
    python demo_kinematics.py --demo                     # no hardware, sine sweep
    python demo_kinematics.py                            # real motors, free motion
    python demo_kinematics.py --port /dev/ttyACM1        # custom port
    python demo_kinematics.py --look-at 0.5 0.0 0.4      # IK aim at a point
    python demo_kinematics.py --no-viz                   # math verification only
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # add head/ to path

import argparse
import numpy as np

from kinematics import fk, fk_position, fk_axis, ik, ik_look_at
from visualizer import run_visualizer, DemoSource, HardwareSource, LookAtSource


def _run_math_verification() -> None:
    print("=" * 68)
    print("  FK / IK 数学验证（无硬件）")
    print("=" * 68)

    print("\n[1] 零位位姿")
    T0  = fk(0, 0)
    pos = fk_position(0, 0)
    ax  = fk_axis(0, 0)
    print(f"  相机原点 : {pos}")
    print(f"  相机光轴 : {ax}")
    print(f"  变换矩阵 :\n{T0}")

    print("\n[2] FK → IK 互逆性（误差应 < 0.01）")
    cases = [
        (   0,    0, "零位"),
        (   0,   50, "pan  +50"),
        (   0,  -50, "pan  -50"),
        (  40,    0, "tilt +40"),
        ( -40,    0, "tilt -40"),
        (  30,   45, "斜向右下"),
        ( -20,  -60, "斜向左上"),
    ]
    hdr = f"  {'说明':<12} {'输入(m1,m2)':<16} {'相机位置 (m)':<38} {'IK还原':<18} {'误差'}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    all_ok = True
    for m1, m2, desc in cases:
        pos  = fk_position(m1, m2)
        axis = fk_axis(m1, m2)
        m1r, m2r = ik(axis)
        err  = max(abs(m1r - m1), abs(m2r - m2))
        ok   = err < 0.01
        all_ok = all_ok and ok
        flag = "✓" if ok else "✗"
        print(f"  {flag} {desc:<12} ({m1:+5.0f},{m2:+5.0f})  "
              f"[{pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}]  "
              f"({m1r:+6.2f},{m2r:+6.2f})  {err:.4f}")
    print(f"\n  结论: {'全部通过 ✓' if all_ok else '存在失败项 ✗'}")

    print("\n[3] IK look-at 验证")
    targets = [
        ([1.0,  0.0,  0.0], "正前方"),
        ([1.0,  1.0,  0.0], "右前 45°"),
        ([1.0,  0.0, -0.5], "前下方"),
        ([0.5, -0.5,  0.3], "左前上"),
    ]
    print(f"  {'目标方向':<14} {'电机(m1,m2)':<18} {'光轴':<34} {'对准误差'}")
    print("  " + "-" * 78)
    for tgt, desc in targets:
        m1, m2 = ik_look_at(tgt)
        axis   = fk_axis(m1, m2)
        tgt_n  = np.array(tgt) / np.linalg.norm(tgt)
        err    = np.linalg.norm(axis - tgt_n)
        print(f"  {desc:<14} ({m1:+6.2f},{m2:+6.2f})  "
              f"[{axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}]  {err:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相机头运动学验证 + 实时三维可视化")
    parser.add_argument("--demo",    action="store_true",
                        help="无硬件 demo 模式（正弦摇摆）")
    parser.add_argument("--port",    default="/dev/ttyACM0",
                        help="电机串口（默认 /dev/ttyACM0）")
    parser.add_argument("--fps",     type=int, default=20,
                        help="可视化刷新帧率（默认 20）")
    parser.add_argument("--no-viz",  action="store_true",
                        help="只跑数学验证，不启动可视化")
    parser.add_argument("--look-at", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        help="IK 对准目标点（top_base_link 坐标系，单位：米）")
    args = parser.parse_args()

    target_point = np.array(args.look_at) if args.look_at else None

    _run_math_verification()

    if target_point is not None:
        m1, m2 = ik_look_at(target_point)
        print(f"\n[look-at] 目标点 {target_point.tolist()}  →  "
              f"motor1={m1:+.2f}  motor2={m2:+.2f}")

    if args.no_viz:
        raise SystemExit(0)

    print("\n按 ENTER 启动可视化窗口（Ctrl-C 退出）...")
    try:
        input()
    except KeyboardInterrupt:
        raise SystemExit(0)

    if target_point is not None:
        source = LookAtSource(target_point, port=args.port)
        try:
            run_visualizer(source, interval_ms=1000 // args.fps,
                           target_point=target_point)
        finally:
            source.close()
            print("电机已断开。")
    elif args.demo:
        run_visualizer(DemoSource(), interval_ms=1000 // args.fps)
    else:
        source = HardwareSource(port=args.port)
        try:
            run_visualizer(source, interval_ms=1000 // args.fps)
        finally:
            source.close()
            print("电机已断开。")
