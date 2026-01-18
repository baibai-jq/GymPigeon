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
            ExerciseType.PUSHUP: ["shoulder", "elbow", "wrist", "hip", "ankle"]
        }
        
        # Form criteria for each exercise
        self.FORM_CRITERIA = self._initialize_form_criteria()
    
    def _initialize_form_criteria(self):
        """Initialize angle criteria for proper form"""
        return {
            ExerciseType.SQUAT: {
                "knee_min": 70,
                "knee_max": 110,
                "hip_min": 30,
                "hip_max": 120,
                "back_min": 100,
                "knee_forward_threshold": 30
            },
            ExerciseType.BENCH_PRESS: {
                "elbow_min": 70,
                "elbow_max": 110,
                "back_arch_min": 150,
                "back_arch_max": 175,
                "arm_angle_min": 45,
                "arm_angle_max": 75,
                "bar_path_deviation": 10
            },
            ExerciseType.PUSHUP: {
                "elbow_min": 70,
                "elbow_max": 110,
                "back_min": 160,
                "back_max": 180,
                "shoulder_stability_min": 135,
                "body_alignment_min": 160
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
        
        # Store shoulder and elbow Y positions FIRST (needed for push-ups)
        if f"{side}_shoulder" in points:
            angles["shoulder_y"] = points[f"{side}_shoulder"][1]
        
        if f"{side}_elbow" in points:
            angles["elbow_y"] = points[f"{side}_elbow"][1]
        
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
        """Detect current phase of the exercise"""
        if exercise == ExerciseType.SQUAT:
            knee_angle = angles.get("knee", 180)
            if knee_angle >= 140:
                return RepPhase.READY
            elif knee_angle <= 120:
                return RepPhase.BOTTOM
            elif self.current_phase == RepPhase.READY:
                return RepPhase.DESCENDING
            else:
                return RepPhase.ASCENDING
        
        elif exercise == ExerciseType.BENCH_PRESS:
            elbow_angle = angles.get("elbow", 180)
            if elbow_angle >= 140:
                return RepPhase.READY
            elif elbow_angle <= 120:
                return RepPhase.BOTTOM
            elif self.current_phase == RepPhase.READY:
                return RepPhase.DESCENDING
            else:
                return RepPhase.ASCENDING

        elif exercise == ExerciseType.PUSHUP:
            # Use shoulder height (Y position) to detect push-up phases
            # In side view during push-up:
            # - READY/TOP: shoulder is HIGH (small Y value) - body elevated
            # - BOTTOM: shoulder is LOW (large Y value) - body near ground
            shoulder_y = angles.get("shoulder_y")
            elbow_angle = angles.get("elbow", 180)
            
            if shoulder_y is not None:
                # Track previous shoulder height for direction detection
                if not hasattr(self, '_last_shoulder_y'):
                    self._last_shoulder_y = shoulder_y
                    self._shoulder_baseline = shoulder_y
                    self._pushup_bottom_reached = False
                
                # Calculate movement direction
                delta_y = shoulder_y - self._last_shoulder_y
                
                # DEBUG: Print the values
                print(f"[PUSHUP DEBUG] Shoulder Y: {shoulder_y:.1f}, Last: {self._last_shoulder_y:.1f}, Delta: {delta_y:.1f}, Elbow: {elbow_angle:.1f}, Current Phase: {self.current_phase.value}")
                
                # State machine for push-up phases
                if self.current_phase == RepPhase.READY:
                    # Update baseline when in ready position
                    if shoulder_y < self._shoulder_baseline:
                        self._shoulder_baseline = shoulder_y
                    
                    # Start descending when shoulder drops significantly
                    if shoulder_y > self._shoulder_baseline + 50:
                        phase = RepPhase.DESCENDING
                        self._pushup_bottom_reached = False
                    else:
                        phase = RepPhase.READY
                
                elif self.current_phase == RepPhase.DESCENDING:
                    # Check if we've reached bottom (shoulder stopped descending)
                    # Bottom is reached when shoulder stops moving down significantly
                    if delta_y < 2 and (shoulder_y > self._shoulder_baseline + 100):
                        # Mark that we've been to bottom
                        self._pushup_bottom_reached = True
                        phase = RepPhase.BOTTOM
                    # Or if shoulder starts moving back up, we're ascending
                    elif delta_y < -5:
                        phase = RepPhase.ASCENDING
                    else:
                        phase = RepPhase.DESCENDING
                
                elif self.current_phase == RepPhase.BOTTOM:
                    # Start ascending when shoulder moves up
                    if delta_y < -5:
                        phase = RepPhase.ASCENDING
                    else:
                        phase = RepPhase.BOTTOM
                
                elif self.current_phase == RepPhase.ASCENDING:
                    # Return to ready when close to baseline
                    if shoulder_y < self._shoulder_baseline + 40:
                        phase = RepPhase.READY
                        self._pushup_bottom_reached = False
                    # If still moving down, back to descending
                    elif delta_y > 5:
                        phase = RepPhase.DESCENDING
                    else:
                        phase = RepPhase.ASCENDING
                
                else:
                    phase = RepPhase.READY
                
                self._last_shoulder_y = shoulder_y
                print(f"[PUSHUP DEBUG] Detected Phase: {phase.value}")
                return phase
            else:
                # Fallback to elbow angle if Y positions not available
                elbow_angle = angles.get("elbow", 180)
                if elbow_angle >= 150:
                    return RepPhase.READY
                elif elbow_angle <= 100:
                    return RepPhase.BOTTOM
                elif self.current_phase == RepPhase.READY or self.current_phase == RepPhase.DESCENDING:
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
        """Evaluate squat form"""
        errors = []
        warnings = []

        criteria = {
            "knee_min": 120,
            "back_min": 35,
        }

        def filter_descent_angles(joint: str):
            angles = rep_data.phase_angles.get(joint, [])
            if not angles:
                return []
            n = len(angles)
            return angles[:max(1, n // 2)]

        descent_knee = filter_descent_angles("knee")
        descent_hip = filter_descent_angles("hip")
        descent_back = filter_descent_angles("back_alignment_smoothed") or filter_descent_angles("back_alignment")
        descent_hip_y = filter_descent_angles("hip_y")
        descent_knee_y = filter_descent_angles("knee_y")

        min_knee = min(descent_knee) if descent_knee else 180
        min_hip = min(descent_hip) if descent_hip else 180
        min_back = min(descent_back) if descent_back else 180

        if min_knee > criteria["knee_min"]:
            warnings.append(f"Shallow squat: Min knee angle {min_knee:.1f}° (target: <{criteria['knee_min']}°)")
        elif min_knee < 65:
            warnings.append(f"Very deep squat: Min knee angle {min_knee:.1f}°")

        if min_back < criteria["back_min"]:
            warnings.append(f"Back rounding: Min back angle {min_back:.1f}° (target: >{criteria['back_min']}°)")

        if min_hip > 95:
            warnings.append(f"Sit back more: Min hip angle {min_hip:.1f}°")
        elif min_hip < 50:
            errors.append(f"Forward lean: Min hip angle {min_hip:.1f}° (target: >{50}°)")

        if descent_hip_y and descent_knee_y:
            hip_bottom = max(descent_hip_y)
            knee_bottom = max(descent_knee_y)
            if hip_bottom < knee_bottom - 50:
                errors.append("Insufficient depth: Hips should be at knee level or up to 50 pixels higher")

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
        
        if min_elbow > 100:
            errors.append(f"Range issue: Min elbow angle {min_elbow:.1f}° (target: ~90°)")
        elif min_elbow < 60:
            warnings.append(f"Deep press: Min elbow angle {min_elbow:.1f}°")
        
        if back_angles:
            avg_back = np.mean(back_angles)
            if avg_back < 140:
                errors.append(f"Excessive arch: Avg back angle {avg_back:.1f}° (target: 140-175°)")
            elif avg_back > 175:
                errors.append(f"Back too flat: Avg back angle {avg_back:.1f}° (target: 140-175°)")
        
        if arm_body < 35:
            errors.append(f"Narrow grip: Min arm angle {arm_body:.1f}° (target: 45-75°)")
        elif arm_body > 85:
            warnings.append(f"Wide grip: Min arm angle {arm_body:.1f}°")
        
        is_correct = len(errors) == 0
        feedback = self._generate_feedback(ExerciseType.BENCH_PRESS, errors, warnings)
        
        return FormEvaluation(is_correct, self.rep_count, errors, warnings, feedback)
    
    def evaluate_pushup(self, rep_data: RepData) -> FormEvaluation:
        """Evaluate push-up form using only angles from READY and DESCENDING phases (rest to end of descent)"""
        errors = []
        warnings = []
        
        # EXTREMELY LOOSE criteria for push-up
        criteria = {
            "elbow_min": 120,    # Minimum elbow bend (was 70, very loose now)
            "hip_min": 150,      # Body alignment (hip angle should stay relatively straight)
        }

        # Only use angles recorded during READY and DESCENDING phases
        def filter_descent_angles(joint: str):
            # rep_data.phase_angles[joint] is a list of angles in order
            # Use first half of the angles (from READY to end of DESCENDING)
            angles = rep_data.phase_angles.get(joint, [])
            if not angles:
                return []
            # Use first half of the angles (from READY to end of DESCENDING)
            n = len(angles)
            return angles[:max(1, n // 2)]

        descent_elbow = filter_descent_angles("elbow")
        descent_hip = filter_descent_angles("hip")
        descent_shoulder = filter_descent_angles("shoulder")

        min_elbow = min(descent_elbow) if descent_elbow else 180
        min_hip = min(descent_hip) if descent_hip else 180
        avg_shoulder = np.mean(descent_shoulder) if descent_shoulder else 180

        # Check elbow bend depth (VERY LOOSE)
        # Only warn if elbow angle is above 120 (very shallow)
        if min_elbow > criteria["elbow_min"]:
            warnings.append(f"Shallow push-up: Min elbow angle {min_elbow:.1f}° (target: <{criteria['elbow_min']}°)")
        elif min_elbow < 65:
            warnings.append(f"Very deep push-up: Min elbow angle {min_elbow:.1f}°")
        
        # Check body alignment (hip angle should stay straight)
        # LOOSE: Only flag if hips sag significantly
        if min_hip < criteria["hip_min"]:
            warnings.append(f"Sagging hips: Min hip angle {min_hip:.1f}° (target: >{criteria['hip_min']}°)")
        
        # Check shoulder stability (LOOSE)
        if avg_shoulder < 130:
            warnings.append(f"Shoulder instability: Avg angle {avg_shoulder:.1f}° (target: >130°)")
        
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
            torso_h = None
            if "right_hip" in points:
                torso_h = abs(points["right_hip"][1] - ((ry+ly)/2))
            if torso_h is None or torso_h <= 0:
                torso_h = 200.0
            if dx > max(50, 0.15 * torso_h):
                if self.shoulder_warning_cooldown <= 0:
                    shoulder_warning = "⚠ You appear rotated – ensure you're directly SIDEWAYS to camera (shoulders overlapping)."
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
            if not isinstance(v, (int, float)):
                continue
            if k.endswith("_y"):
                self.last_angles[k] = v
                continue
            last = self.last_angles.get(k)
            if last is not None:
                if abs(v - last) > self.angle_jump_threshold:
                    angles[k] = last
                else:
                    self.last_angles[k] = v
            else:
                self.last_angles[k] = v

        # Detect phase
        new_phase = self.detect_phase(angles, exercise)

        # Handle bottom-hold persistence to avoid bounce counting
        if new_phase == RepPhase.BOTTOM:
            self.bottom_hold_counter += 1
            if self.bottom_hold_counter < self.bottom_hold_required:
                new_phase = RepPhase.DESCENDING if self.current_phase in (RepPhase.READY, RepPhase.DESCENDING) else self.current_phase
            else:
                new_phase = RepPhase.BOTTOM
        else:
            self.bottom_hold_counter = 0

        # DEBUG: Phase transition logging
        if exercise == ExerciseType.PUSHUP:
            print(f"[PHASE] Last: {self.last_phase.value} -> Current: {self.current_phase.value} -> New: {new_phase.value}")

        # Phase transition: start rep when transitioning from READY to DESCENDING
        if self.last_phase == RepPhase.READY and new_phase == RepPhase.DESCENDING:
            self.start_rep(exercise, timestamp)
            print(f"[REP START] Started rep at timestamp {timestamp}")

        # Record angles during rep
        if self.current_rep:
            self.record_angles(angles)

        # Complete rep when returning to READY from ASCENDING
        evaluation = None
        if self.last_phase == RepPhase.ASCENDING and new_phase == RepPhase.READY:
            if self.current_rep:
                rep_data = self.complete_rep(timestamp)
                print(f"[REP COMPLETE] Rep #{self.rep_count} completed!")

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

        return evaluation, shoulder_warning
    
    def reset(self):
        """Reset all tracking data"""
        self.current_phase = RepPhase.READY
        self.last_phase = RepPhase.READY
        self.rep_count = 0
        self.current_rep = None
        self.completed_reps = []
        self.last_angles = {}
        self.bottom_hold_counter = 0


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