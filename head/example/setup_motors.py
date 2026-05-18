"""
One-time motor ID assignment.

Connect each motor alone to the bus when prompted, then run this script.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))  # add head/ to path

from head_motor import HeadMotorController

PORT = "/dev/ttyACM0"

if __name__ == "__main__":
    head = HeadMotorController(port=PORT)
    head.bus.connect()
    try:
        head.setup_motors()
        print("Motor ID setup complete.")
    finally:
        head.bus.disconnect(True)
