"""
Basic head motor control demo.

Sweeps motor1 (tilt) and motor2 (pan) through their limits and prints positions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # add head/ to path

import time
from head_motor import HeadMotorController

PORT = "/dev/ttyACM1"

head = HeadMotorController(port=PORT)
head.connect()

try:
    print(f"Current position: {head.get_pos()}")

    for m1, m2, label in [
        ( 100.0,    0.0, "tilt max"),
        (-100.0,    0.0, "tilt min"),
        (   0.0,  100.0, "pan  max"),
        (   0.0, -100.0, "pan  min"),
        (   0.0,    0.0, "centre  "),
    ]:
        head.send_pos(m1, m2)
        time.sleep(0.5)
        print(f"  [{label}] {head.get_pos()}")
        time.sleep(0.5)

finally:
    head.disconnect()
