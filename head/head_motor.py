import json
import logging
from dataclasses import asdict
from pathlib import Path

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

logger = logging.getLogger(__name__)

CALIBRATION_FILE = Path("head_calibration.json")


class HeadMotorController:
    """
    Standalone controller for the 2-DOF observation head on XLeRobot.
      head_motor_1 (ID 7): pan / yaw
      head_motor_2 (ID 8): tilt / pitch
    Both motors share one Feetech bus (default /dev/ttyACM0).
    Positions are in the normalized range [-100, 100].
    """

    MOTORS = ["head_motor_1", "head_motor_2"]

    def __init__(self, port: str = "/dev/ttyACM0",
                 calibration_file: Path = CALIBRATION_FILE):
        self.calibration_file = Path(calibration_file)
        self.bus = FeetechMotorsBus(
            port=port,
            motors={
                "head_motor_1": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
                "head_motor_2": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            },
        )

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if self.calibration_file.is_file():
            user_input = input(
                "Calibration file found. Press ENTER to restore, or 'c' for manual calibration: "
            )
            if user_input.strip().lower() != "c":
                self._restore_calibration()
            elif calibrate:
                self.calibrate()
        elif calibrate:
            self.calibrate()
        self._configure()
        logger.info("Head motors connected.")

    def calibrate(self) -> None:
        self.bus.disable_torque()
        for name in self.MOTORS:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move head motors to middle of their range of motion, then press ENTER...")
        homing_offsets = self.bus.set_half_turn_homings(self.MOTORS)

        print("Move head through full range of motion. Press ENTER to stop...")
        range_mins, range_maxes = self.bus.record_ranges_of_motion(self.MOTORS)

        calibration = {
            name: MotorCalibration(
                id=self.bus.motors[name].id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )
            for name in self.MOTORS
        }
        self.bus.calibration = calibration
        self.bus.write_calibration(calibration)
        self._save_calibration(calibration)
        print(f"Calibration saved to {self.calibration_file}")

    def _configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.MOTORS:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            self.bus.write("P_Coefficient", name, 16)
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 43)
        self.bus.enable_torque()

    def get_pos(self) -> dict[str, float]:
        """Read current positions: {"head_motor_1": float, "head_motor_2": float}"""
        return self.bus.sync_read("Present_Position", self.MOTORS)

    def send_pos(self, head_motor_1: float, head_motor_2: float) -> None:
        """Send goal positions (range -100 to 100)."""
        self.bus.sync_write("Goal_Position", {
            "head_motor_1": head_motor_1,
            "head_motor_2": head_motor_2,
        })

    def disconnect(self) -> None:
        self.bus.disconnect(True)
        logger.info("Head motors disconnected.")

    def setup_motors(self) -> None:
        """Assign motor IDs one by one. Connect each motor alone to the bus when prompted."""
        for name in reversed(self.MOTORS):
            input(f"Connect the controller board to the '{name}' motor only and press ENTER.")
            self.bus.setup_motor(name)
            print(f"'{name}' motor ID set to {self.bus.motors[name].id}")

    def _restore_calibration(self) -> None:
        try:
            with open(self.calibration_file) as f:
                data = json.load(f)
            calibration = {
                name: MotorCalibration(**fields)
                for name, fields in data.items()
                if name in self.MOTORS
            }
            self.bus.calibration = calibration
            self.bus.write_calibration(calibration)
            logger.info("Calibration restored from file.")
        except Exception as e:
            logger.warning(f"Failed to load calibration: {e}. Running manual calibration...")
            self.calibrate()

    def _save_calibration(self, calibration: dict) -> None:
        self.calibration_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.calibration_file, "w") as f:
            json.dump({name: asdict(cal) for name, cal in calibration.items()}, f, indent=2)
