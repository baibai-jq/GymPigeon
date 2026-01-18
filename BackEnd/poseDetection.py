"""
Pose Detection Module for Gym Exercise Form Correction
Handles: Squats, Bench Press, Push-ups (all from side view - 90 degrees)
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ExerciseType(Enum):
    SQUAT = "squat"
    BENCH_PRESS = "bench_press"
    PUSHUP = "pushup"

class RepPhase(Enum):
    READY = "ready"
    DESCENDING = "descending"
    BOTTOM = "bottom"
    ASCENDING = "ascending"

@dataclass
class RepData:
    """Stores data for a single rep"""
    exercise: ExerciseType
    phase_angles: Dict[str, List[float]]  # {joint_name: [angles throughout rep]}
    start_time: float
    end_time: Optional[float] = None
    max_angles: Dict[str, float] = None
    min_angles: Dict[str, float] = None
    
    def __post_init__(self):
        self.max_angles = {}
        self.min_angles = {}

@dataclass
class FormEvaluation:
    """Result of form evaluation"""
    is_correct: bool
    rep_number: int
    errors: List[str]
    warnings: List[str]
    feedback_message: str

class PoseDetection:
    def __init__(self):
        self.current_exercise: Optional[ExerciseType] = None
        self.current_phase = RepPhase.READY
        self.last_phase = RepPhase.READY
        self.rep_count = 0
        self.current_rep: Optional[RepData] = None
        self.completed_reps: List[RepData] = []
        # Smoothing window for back alignment (median filter)
        self.back_window = deque(maxlen=7)  # keep last 7 frames
        self.window_size = 7

        # Last recorded angles to detect impossible jumps
        self.last_angles: Dict[str, float] = {}
        self.angle_jump_threshold = 30.0  # degrees per frame considered impossible

        # Bottom hold counters to require holding bottom for a few frames
        self.bottom_hold_counter = 0
        self.bottom_hold_required = 3

        # Shoulder side-view warning cooldown to avoid spamming
        self.shoulder_warning_cooldown = 0
        
        # Define required landmarks for each exercise (side view)
        self.REQUIRED_LANDMARKS = {
            ExerciseType.SQUAT: ["shoulder", "hip", "knee", "ankle"],
            ExerciseType.BENCH_PRESS: ["shoulder", "elbow", "wrist", "hip"],
            ExerciseType.PUSHUP: ["shoulder", "elbow", "wrist", "hip", "knee"]
        }
        
        # Form criteria for each exercise
        self.FORM_CRITERIA = self._initialize_form_criteria()
    
    def _initialize_form_criteria(self):
        """Initialize angle criteria for proper form"""
        return {
            ExerciseType.SQUAT: {
                "knee_min": 70,    # Minimum knee bend at bottom
                "knee_max": 110,   # Maximum knee bend (too deep)
                "hip_min": 60,     # Minimum hip angle (sitting back)
                "hip_max": 100,    # Maximum hip angle at bottom
                "back_min": 140,   # Back straightness (shoulder-hip-knee)
                "knee_forward_threshold": 20  # Knee shouldn't go too far forward
            },
            ExerciseType.BENCH_PRESS: {
                "elbow_min": 70,   # Minimum elbow bend
                "elbow_max": 110,  # Maximum elbow bend at bottom
                "back_arch_min": 150,  # Slight arch in back
                "back_arch_max": 175,
                "arm_angle_min": 45,   # Arm angle from body (not too wide)
                "arm_angle_max": 75,
                "bar_path_deviation": 10  # Maximum horizontal deviation
            },
            ExerciseType.PUSHUP: {
                "elbow_min": 70,   # Minimum elbow bend
                "elbow_max": 110,  # Maximum elbow bend at bottom
                "back_min": 160,   # Back should be straight (hip angle)
                "back_max": 180,
                "shoulder_stability_min": 150,  # Shoulders shouldn't sag
                "body_alignment_min": 160  # Overall body alignment
            }
        }
    
    def calculate_angle(self, a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        """Calculate angle at point B given points A, B, C"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        
        return angle
    
    def check_required_landmarks(self, points: Dict[str, Tuple[int, int]], 
                                 exercise: ExerciseType) -> Tuple[bool, List[str]]:
        """
        Check if all required landmarks are visible for the exercise
        Returns: (all_present, missing_landmarks)
        """
        required = self.REQUIRED_LANDMARKS[exercise]
        
        # For side view, check if at least one complete side is visible
        right_side = [f"right_{lm}" for lm in required]
        left_side = [f"left_{lm}" for lm in required]
        
        right_complete = all(lm in points for lm in right_side)
        left_complete = all(lm in points for lm in left_side)
        
        if right_complete or left_complete:
            return True, []
        
        # Find missing landmarks
        right_missing = [lm.replace("right_", "") for lm in right_side if lm not in points]
        left_missing = [lm.replace("left_", "") for lm in left_side if lm not in points]
        
        # Return the side with fewer missing landmarks
        if len(right_missing) <= len(left_missing):
            return False, right_missing
        else:
            return False, left_missing
    
    def get_repositioning_message(self, exercise: ExerciseType, missing: List[str]) -> str:
        """Generate warning message for repositioning"""
        missing_str = ", ".join([m.replace("_", " ").title() for m in missing])
        
        messages = {
            ExerciseType.SQUAT: f"⚠ Cannot detect {missing_str}. Stand SIDEWAYS to camera. Ensure your full profile is visible from shoulder to ankle.",
            ExerciseType.BENCH_PRESS: f"⚠ Cannot detect {missing_str}. Position camera to see your SIDE VIEW on the bench. Entire arm and torso should be visible.",
            ExerciseType.PUSHUP: f"⚠ Cannot detect {missing_str}. Position camera to see your SIDE PROFILE. Full body from head to feet should be visible."
        }
        
        return messages.get(exercise, "⚠ Reposition to show side view of your body.")
    
    def extract_angles(self, points: Dict[str, Tuple[int, int]]) -> Dict[str, float]:
        """Extract all relevant angles from detected landmarks"""
        angles = {}
        
        # Determine which side is visible (prefer right, fall back to left)
        side = "right" if f"right_shoulder" in points else "left"
        
        # Knee angle
        if all(f"{side}_{lm}" in points for lm in ["hip", "knee", "ankle"]):
            angles["knee"] = self.calculate_angle(
                points[f"{side}_hip"],
                points[f"{side}_knee"],
                points[f"{side}_ankle"]
            )
            # record raw Y positions for relative depth checks (image coords: larger Y is lower)
            angles["hip_y"] = points[f"{side}_hip"][1]
            angles["knee_y"] = points[f"{side}_knee"][1]
        
        # Hip angle (shoulder-hip-knee)
        if all(f"{side}_{lm}" in points for lm in ["shoulder", "hip", "knee"]):
            angles["hip"] = self.calculate_angle(
                points[f"{side}_shoulder"],
                points[f"{side}_hip"],
                points[f"{side}_knee"]
            )
        
        # Elbow angle
        if all(f"{side}_{lm}" in points for lm in ["shoulder", "elbow", "wrist"]):
            angles["elbow"] = self.calculate_angle(
                points[f"{side}_shoulder"],
                points[f"{side}_elbow"],
                points[f"{side}_wrist"]
            )
        
        # Shoulder angle (for back position)
        if all(f"{side}_{lm}" in points for lm in ["elbow", "shoulder", "hip"]):
            angles["shoulder"] = self.calculate_angle(
                points[f"{side}_elbow"],
                points[f"{side}_shoulder"],
                points[f"{side}_hip"]
            )
        
        # Back/body alignment (shoulder-hip-ankle for overall straightness)
        if all(f"{side}_{lm}" in points for lm in ["shoulder", "hip", "ankle"]):
            angles["back_alignment"] = self.calculate_angle(
                points[f"{side}_shoulder"],
                points[f"{side}_hip"],
                points[f"{side}_ankle"]
            )
        
        # Arm angle from body (for bench press)
        if all(f"{side}_{lm}" in points for lm in ["hip", "shoulder", "elbow"]):
            angles["arm_body_angle"] = self.calculate_angle(
                points[f"{side}_hip"],
                points[f"{side}_shoulder"],
                points[f"{side}_elbow"]
            )
        
        return angles
    
    def detect_phase(self, angles: Dict[str, float], exercise: ExerciseType) -> RepPhase:
        """Detect current phase of the exercise based on joint angles"""
        if exercise == ExerciseType.SQUAT:
            knee_angle = angles.get("knee", 180)
            if knee_angle >= 140:
                return RepPhase.READY
            elif knee_angle <= 120:
                return RepPhase.BOTTOM
            elif self.current_phase in [RepPhase.READY, RepPhase.DESCENDING]:
                return RepPhase.DESCENDING
            else:
                return RepPhase.ASCENDING
        
        elif exercise in [ExerciseType.BENCH_PRESS, ExerciseType.PUSHUP]:
            elbow_angle = angles.get("elbow", 180)
            if elbow_angle >= 140:
                return RepPhase.READY
            elif elbow_angle <= 120:
                return RepPhase.BOTTOM
            elif self.current_phase in [RepPhase.READY, RepPhase.DESCENDING]:
                return RepPhase.DESCENDING
            else:
                return RepPhase.ASCENDING
        
        return self.current_phase
    
    def start_rep(self, exercise: ExerciseType, timestamp: float):
        """Start tracking a new rep"""
        self.current_rep = RepData(
            exercise=exercise,
            phase_angles={},
            start_time=timestamp
        )
    
    def record_angles(self, angles: Dict[str, float]):
        """Record angles for current rep"""
        if not self.current_rep:
            return
        
        for joint, angle in angles.items():
            if joint not in self.current_rep.phase_angles:
                self.current_rep.phase_angles[joint] = []
            self.current_rep.phase_angles[joint].append(angle)
    
    def complete_rep(self, timestamp: float) -> RepData:
        """Complete current rep and calculate min/max angles"""
        if not self.current_rep:
            return None
        
        self.current_rep.end_time = timestamp
        
        # Calculate min and max angles for each joint
        for joint, angle_list in self.current_rep.phase_angles.items():
            self.current_rep.min_angles[joint] = min(angle_list)
            self.current_rep.max_angles[joint] = max(angle_list)
        
        completed = self.current_rep
        self.completed_reps.append(completed)
        self.current_rep = None
        self.rep_count += 1
        
        return completed
    
    def evaluate_squat(self, rep_data: RepData) -> FormEvaluation:
        """Evaluate squat form based on collected rep data"""
        errors = []
        warnings = []
        
        criteria = self.FORM_CRITERIA[ExerciseType.SQUAT]
        min_knee = rep_data.min_angles.get("knee", 180)
        min_hip = rep_data.min_angles.get("hip", 180)
        back_list = rep_data.phase_angles.get("back_alignment_smoothed", []) or rep_data.phase_angles.get("back_alignment", [])
        min_back = min(back_list) if back_list else rep_data.min_angles.get("back_alignment", 180)
        
        # Check squat depth (lower angle = deeper squat)
        if min_knee > 100:
            errors.append(f"Depth issue: Min knee angle {min_knee:.1f}° (target: <90°)")
        elif min_knee < 65:
            warnings.append(f"Very deep squat: Min knee angle {min_knee:.1f}°")
        
        # Check hip hinge and back angle
        if min_back < criteria["back_min"]:
            errors.append(f"Back rounding: Min back angle {min_back:.1f}° (target: >{criteria['back_min']}°)")
        
        # Check hip position relative to knees
        if min_hip > 95:
            warnings.append(f"Sit back more: Min hip angle {min_hip:.1f}°")
        elif min_hip < 50:
            errors.append(f"Forward lean: Min hip angle {min_hip:.1f}° (target: >{50}°)")

        # Relative depth check using Y coordinates collected during rep
        hip_ys = rep_data.phase_angles.get("hip_y", [])
        knee_ys = rep_data.phase_angles.get("knee_y", [])
        if hip_ys and knee_ys:
            hip_bottom = max(hip_ys)
            knee_bottom = max(knee_ys)
            if hip_bottom < knee_bottom - 10:
                errors.append("Insufficient depth: Hips below knee required")
        
        is_correct = len(errors) == 0
        feedback = self._generate_feedback(ExerciseType.SQUAT, errors, warnings)
        
        return FormEvaluation(is_correct, self.rep_count, errors, warnings, feedback)
    
    def evaluate_bench_press(self, rep_data: RepData) -> FormEvaluation:
        """Evaluate bench press form based on collected rep data"""
        errors = []
        warnings = []
        
        criteria = self.FORM_CRITERIA[ExerciseType.BENCH_PRESS]
        min_elbow = rep_data.min_angles.get("elbow", 180)
        back_angles = rep_data.phase_angles.get("back_alignment", [])
        arm_body = rep_data.min_angles.get("arm_body_angle", 90)
        
        # Check elbow bend depth
        if min_elbow > criteria["elbow_max"]:
            errors.append(f"Lower bar to chest level") # ({criteria['elbow_min']}-{criteria['elbow_max']}° elbow, achieved {min_elbow:.0f}°)
        elif min_elbow < criteria["elbow_min"]:
            warnings.append(f"Very deep press")
        
        # Check back stability
        if back_angles:
            avg_back = np.mean(back_angles)
            if avg_back < 140:
                errors.append(f"Excessive arch: Avg back angle {avg_back:.1f}° (target: 140-175°)")
            elif avg_back > 175:
                errors.append(f"Back too flat: Avg back angle {avg_back:.1f}° (target: 140-175°)")
        
        # Check arm angle from body (not too wide)
        if arm_body < criteria["arm_angle_min"]:
            errors.append(f"Arms too close to body: Widen your grip") # (arm angle {arm_body:.0f}°)
        elif arm_body > criteria["arm_angle_max"]:
            errors.append(f"Arms too wide: Narrow your grip")
        
        # Check bar path (wrist should move vertically)
        if "wrist" in rep_data.phase_angles:
            # This would require tracking horizontal movement - simplified for now
            warnings.append("Ensure bar path is straight up and down")
        
        is_correct = len(errors) == 0
        feedback = self._generate_feedback(ExerciseType.BENCH_PRESS, errors, warnings)
        
        return FormEvaluation(is_correct, self.rep_count, errors, warnings, feedback)
    
    def evaluate_pushup(self, rep_data: RepData) -> FormEvaluation:
        """Evaluate push-up form based on collected rep data"""
        errors = []
        warnings = []
        
        criteria = self.FORM_CRITERIA[ExerciseType.PUSHUP]
        min_elbow = rep_data.min_angles.get("elbow", 180)
        hip_angles = rep_data.phase_angles.get("hip", [])
        shoulder_angles = rep_data.phase_angles.get("shoulder", [])
        
        # Check elbow bend depth
        if min_elbow > criteria["elbow_max"]:
            errors.append(f"Lower chest to ground") # ({criteria['elbow_min']}-{criteria['elbow_max']}° elbow, achieved {min_elbow:.0f}°)
        elif min_elbow < criteria["elbow_min"]:
            warnings.append(f"Very deep push-up")
        
        # Check back/body alignment (hip angle should stay relatively straight)
        if back_angles:
            avg_back = np.mean(back_angles)
            if avg_back < criteria["back_min"]:
                errors.append(f"Keep core tigh") # (hip angle {avg_back:.0f}°, target {criteria['back_min']}°+)
            elif avg_back > criteria["back_max"]:
                errors.append(f"Lower hips for straight line")
        
        # Check shoulder stability
        if shoulder_angles:
            avg_shoulder = np.mean(shoulder_angles)
            if avg_shoulder < criteria["shoulder_stability_min"]:
                errors.append(f"Keep shoulder blades retracted") # (detected {avg_shoulder:.0f}°)
        
        # Check arm placement (should be roughly shoulder-width)
        warnings.append("Ensure hands are shoulder-width apart")
        
        is_correct = len(errors) == 0
        feedback = self._generate_feedback(ExerciseType.PUSHUP, errors, warnings)
        
        return FormEvaluation(is_correct, self.rep_count, errors, warnings, feedback)
    
    def _generate_feedback(self, exercise: ExerciseType, errors: List[str], warnings: List[str]) -> str:
        """Generate comprehensive feedback message"""
        if not errors and not warnings:
            return f"✓ Excellent {exercise.value.replace('_', ' ')}! Perfect form."
        
        feedback_parts = []
        
        if errors:
            feedback_parts.append(f"Form issues detected in {exercise.value.replace('_', ' ')}:")
            for i, error in enumerate(errors, 1):
                feedback_parts.append(f"{i}. {error}")
        
        if warnings:
            feedback_parts.append("\nAdditional tips:")
            for warning in warnings:
                feedback_parts.append(f"- {warning}")
        
        return "\n".join(feedback_parts)
    
    def process_frame(self, points: Dict[str, Tuple[int, int]], 
                     exercise: ExerciseType, 
                     timestamp: float) -> Tuple[Optional[FormEvaluation], Optional[str]]:
        """
        Main processing function called for each frame
        
        Returns: (evaluation_result, warning_message)
        - evaluation_result: FormEvaluation if rep completed, None otherwise
        - warning_message: Warning if landmarks missing, None otherwise
        """
        # Auto-reset if exercise has changed
        if self.current_exercise != exercise:
            self.reset()
        
        self.current_exercise = exercise
        
        # Check required landmarks
        landmarks_ok, missing = self.check_required_landmarks(points, exercise)
        if not landmarks_ok:
            warning = self.get_repositioning_message(exercise, missing)
            return None, warning
        
        # Extract angles and some raw coordinates
        angles = self.extract_angles(points)

        # Side-view shoulder check (if both shoulders present)
        shoulder_warning = None
        if "right_shoulder" in points and "left_shoulder" in points:
            rx, ry = points["right_shoulder"]
            lx, ly = points["left_shoulder"]
            dx = abs(rx - lx)
            # approximate torso height for scale (if hip present)
            torso_h = None
            if "right_hip" in points:
                torso_h = abs(points["right_hip"][1] - ((ry+ly)/2))
            if torso_h is None or torso_h <= 0:
                torso_h = 200.0
            # If shoulders are separated too much horizontally, user isn't perfectly sideways
            if dx > max(50, 0.15 * torso_h):
                # throttle warnings to reduce spam
                if self.shoulder_warning_cooldown <= 0:
                    shoulder_warning = "⚠ You appear rotated — ensure you're directly SIDEWAYS to camera (shoulders overlapping)."
                    self.shoulder_warning_cooldown = 30

        # Smooth back alignment using median of recent frames
        if "back_alignment" in angles:
            try:
                self.back_window.append(float(angles["back_alignment"]))
                smoothed_back = float(np.median(np.array(self.back_window)))
                angles["back_alignment_smoothed"] = smoothed_back
            except Exception:
                angles["back_alignment_smoothed"] = angles["back_alignment"]

        # Anatomical jitter filter: ignore impossible angle jumps
        for k, v in list(angles.items()):
            # only apply to angle-like keys
            if not isinstance(v, (int, float)):
                continue
            if k.endswith("_y"):
                # allow raw coords to vary freely
                self.last_angles[k] = v
                continue
            last = self.last_angles.get(k)
            if last is not None:
                if abs(v - last) > self.angle_jump_threshold:
                    # treat as jitter: keep last reliable value instead of sudden jump
                    angles[k] = last
                else:
                    self.last_angles[k] = v
            else:
                self.last_angles[k] = v

        # Preliminary phase detection from angles
        prelim_phase = self.detect_phase(angles, exercise)

        # Handle bottom-hold persistence to avoid bounce counting
        new_phase = prelim_phase
        if prelim_phase == RepPhase.BOTTOM:
            self.bottom_hold_counter += 1
            if self.bottom_hold_counter < self.bottom_hold_required:
                # still qualifying as descending until held sufficiently
                new_phase = RepPhase.DESCENDING if self.current_phase in (RepPhase.READY, RepPhase.DESCENDING) else self.current_phase
            else:
                new_phase = RepPhase.BOTTOM
        else:
            self.bottom_hold_counter = 0

        # Relative depth check for squats: ensure hip drops below knee (image coords increase downward)
        if exercise == ExerciseType.SQUAT and "hip_y" in angles and "knee_y" in angles:
            hip_bottom = max(self.last_angles.get("hip_y", angles["hip_y"]), angles["hip_y"])
            knee_at_bottom = max(self.last_angles.get("knee_y", angles["knee_y"]), angles["knee_y"])
            # require hip to be meaningfully below knee to count as bottom
            if new_phase == RepPhase.BOTTOM and hip_bottom <= knee_at_bottom + 5:
                # Not actually below knee; treat as descending (not true bottom)
                new_phase = RepPhase.DESCENDING

        # Phase transition: start rep when transitioning from READY to DESCENDING
        if self.last_phase == RepPhase.READY and new_phase == RepPhase.DESCENDING:
            self.start_rep(exercise, timestamp)

        # Record angles during rep (if any)
        if self.current_rep:
            self.record_angles(angles)

        # Complete rep when returning to READY from ASCENDING
        evaluation = None
        if self.last_phase == RepPhase.ASCENDING and new_phase == RepPhase.READY:
            if self.current_rep:
                rep_data = self.complete_rep(timestamp)

                # Evaluate the completed rep
                if exercise == ExerciseType.SQUAT:
                    evaluation = self.evaluate_squat(rep_data)
                elif exercise == ExerciseType.BENCH_PRESS:
                    evaluation = self.evaluate_bench_press(rep_data)
                elif exercise == ExerciseType.PUSHUP:
                    evaluation = self.evaluate_pushup(rep_data)

        # Decrease shoulder warning cooldown
        if self.shoulder_warning_cooldown > 0:
            self.shoulder_warning_cooldown -= 1

        # Update phase tracking
        self.last_phase = self.current_phase
        self.current_phase = new_phase

        # Prefer returning shoulder rotation warning if any
        return evaluation, shoulder_warning
    
    def reset(self):
        """Reset all tracking data"""
        self.current_phase = RepPhase.READY
        self.last_phase = RepPhase.READY
        self.rep_count = 0
        self.current_rep = None
        self.completed_reps = []
        self.back_window.clear()
        self.last_angles.clear()
        self.bottom_hold_counter = 0
        self.shoulder_warning_cooldown = 0


# Example usage
if __name__ == "__main__":
    detector = PoseDetection()
    
    # Simulate frame processing
    test_points = {
        "right_shoulder": (200, 150),
        "right_elbow": (210, 250),
        "right_wrist": (215, 350),
        "right_hip": (190, 350),
        "right_knee": (185, 500),
        "right_ankle": (180, 650),
    }
    
    # Process frames for a squat
    print("=== Simulating Squat ===")
    
    # Frame 1: Standing
    evaluation, warning = detector.process_frame(test_points, ExerciseType.SQUAT, 0.0)
    if warning:
        print(warning)
    
    # Simulate missing landmarks
    incomplete_points = {"right_shoulder": (200, 150)}
    evaluation, warning = detector.process_frame(incomplete_points, ExerciseType.SQUAT, 0.1)
    if warning:
        print(f"\n{warning}")