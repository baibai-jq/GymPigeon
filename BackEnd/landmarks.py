from dataclasses import dataclass
import os
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np

# ----------------------------
# DATA STRUCTURES
# ----------------------------
@dataclass
class Point:
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class KeyLandmarks:
    shoulder: Point
    elbow: Point
    wrist: Point
    hip: Point
    knee: Point
    ankle: Point


# ----------------------------
# POSE TRACKER CLASS
# ----------------------------
class PoseTracker:
    def __init__(self):
        # Always load model relative to this file
        model_path = os.path.join(
            os.path.dirname(__file__),
            "pose_landmarker_full.task"
        )

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.6,
            min_pose_presence_confidence=0.6,
            min_tracking_confidence=0.5
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(options)

        # Right-side landmark indices
        self.RIGHT = {
            "shoulder": 12,
            "elbow": 14,
            "wrist": 16,
            "hip": 24,
            "knee": 26,
            "ankle": 28,
        }

        # Left-side landmark indices
        self.LEFT = {
            "shoulder": 11,
            "elbow": 13,
            "wrist": 15,
            "hip": 23,
            "knee": 25,
            "ankle": 27,
        }

    def extract_key_landmarks(self, result, side="right") -> KeyLandmarks | None:
        """Extract key landmarks for the given side"""
        if not result.pose_landmarks:
            return None

        lm = result.pose_landmarks[0]  # first detected person
        indices = self.RIGHT if side.lower() == "right" else self.LEFT

        return KeyLandmarks(
            shoulder=Point(lm[indices["shoulder"]].x, lm[indices["shoulder"]].y, lm[indices["shoulder"]].visibility),
            elbow=Point(lm[indices["elbow"]].x, lm[indices["elbow"]].y, lm[indices["elbow"]].visibility),
            wrist=Point(lm[indices["wrist"]].x, lm[indices["wrist"]].y, lm[indices["wrist"]].visibility),
            hip=Point(lm[indices["hip"]].x, lm[indices["hip"]].y, lm[indices["hip"]].visibility),
            knee=Point(lm[indices["knee"]].x, lm[indices["knee"]].y, lm[indices["knee"]].visibility),
            ankle=Point(lm[indices["ankle"]].x, lm[indices["ankle"]].y, lm[indices["ankle"]].visibility),
        )
    import numpy as np

# Inside your PoseTracker class:
    def calculate_angle(self, a, b, c):
        """
        Calculates the angle at point B given points A, B, and C.
        Returns the angle in degrees.
        """
        a = np.array([a.x, a.y]) # First point (e.g., Shoulder)
        b = np.array([b.x, b.y]) # Mid point (e.g., Elbow)
        c = np.array([c.x, c.y]) # End point (e.g., Wrist)

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle


def print_visible_landmarks(key: KeyLandmarks, side=""):
    for name, p in key.__dict__.items():
        if p.visibility >= 0.5:
            print(f"{side} {name}: x={p.x:.2f}, y={p.y:.2f}, vis={p.visibility:.2f}")
