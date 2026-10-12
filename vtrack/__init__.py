"""Vision tracking for robot control: landmarks to joint angles, gestures, head pose, and actuator commands."""
from .filters import LandmarkSmoother, OneEuroFilter
from .gestures import Gesture, GestureDebouncer, GestureFilter, classify_hand, finger_states
from .geometry import (arm_angles, body_frame, finger_flexion, joint_angle, unit)
from .head import HeadPose, HeadPoseEstimator
from .kinematics import ArmAngles, ArmKinematics, frame_agreement_deg
from .mapping import JointChannel, SlewLimiter

__all__ = [
    "ArmAngles", "ArmKinematics", "frame_agreement_deg", "Gesture", "GestureDebouncer", "GestureFilter", "HeadPose",
    "HeadPoseEstimator", "JointChannel", "LandmarkSmoother", "OneEuroFilter", "SlewLimiter", "arm_angles", "body_frame", "classify_hand", "finger_flexion",
    "finger_states", "joint_angle", "unit",
]
