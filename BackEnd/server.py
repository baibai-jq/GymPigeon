import cv2
import mediapipe as mp
import time
import base64
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from landmarks import PoseTracker
from poseDetection import PoseDetection, ExerciseType, FormEvaluation
import numpy as np

from text_to_speech import speak

app = FastAPI()

def extract_points_from_result(result, tracker, h, w):
    """Extract landmark points from MediaPipe result"""
    if not result.pose_landmarks:
        return {}
    
    lm = result.pose_landmarks[0]
    points = {}
    
    # Right side landmarks
    for name, idx in tracker.RIGHT.items():
        if lm[idx].visibility >= 0.5:
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            points[f"right_{name}"] = (cx, cy)
    
    # Left side landmarks
    for name, idx in tracker.LEFT.items():
        if lm[idx].visibility >= 0.5:
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            points[f"left_{name}"] = (cx, cy)
    
    return points

def draw_skeleton_with_feedback(frame, points, warning_message, evaluation, exercise_type, rep_count, current_phase):
    """Draw skeleton with visual feedback"""
    h, w, _ = frame.shape
    
    # If there's a warning (missing landmarks), show it prominently
    if warning_message:
        # Draw dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Warning text
        cv2.putText(frame, "REPOSITION NEEDED", (w//2 - 200, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 3)
        
        # Wrap warning message
        words = warning_message.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + word) < 50:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        
        y_pos = 80
        for line in lines[:3]:  # Max 3 lines
            cv2.putText(frame, line.strip(), (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_pos += 30
        
        # Draw detected points in gray
        for pos in points.values():
            cv2.circle(frame, pos, 6, (100, 100, 100), -1)
        
        return
    
    # Determine skeleton color based on evaluation
    if evaluation and not evaluation.is_correct:
        base_color = (0, 0, 255)  # Red for bad form
    else:
        base_color = (0, 255, 0)  # Green for good form
    
    # Draw skeleton points
    for name, pos in points.items():
        cv2.circle(frame, pos, 8, base_color, -1)
        cv2.circle(frame, pos, 8, (255, 255, 255), 2)  # White outline
    
    # Define skeleton connections
    connections = [
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
    ]
    
    # Draw connections
    for point_a, point_b in connections:
        if point_a in points and point_b in points:
            cv2.line(frame, points[point_a], points[point_b], base_color, 4)
    
    # Display exercise info
    y_offset = 30
    exercise_name = exercise_type.value.replace("_", " ").upper()
    cv2.putText(frame, f"Exercise: {exercise_name}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 35
    
    cv2.putText(frame, f"Reps: {rep_count}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 35
    
    # Phase indicator
    phase_text = current_phase.value.upper()
    phase_color = (0, 255, 255)
    cv2.putText(frame, f"Phase: {phase_text}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, phase_color, 2)
    y_offset += 40
    
    # Show evaluation feedback if available
    if evaluation:
        if evaluation.is_correct:
            status_text = "GOOD FORM ✓"
            status_color = (0, 255, 0)
        else:
            status_text = "CHECK FORM ✗"
            status_color = (0, 0, 255)
        
        cv2.putText(frame, status_text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
        y_offset += 45
        
        # Display errors
        if evaluation.errors:
            cv2.putText(frame, "Issues:", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
            
            for i, error in enumerate(evaluation.errors[:3]):  # Show max 3 errors
                # Truncate long errors
                if len(error) > 60:
                    error = error[:57] + "..."
                cv2.putText(frame, f"- {error}", (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                y_offset += 22

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize Camera, Tracker, and Detector
    cap = cv2.VideoCapture(0)
    tracker = PoseTracker()
    detector = PoseDetection()
    start_time = time.time()
    
    # Default exercise - can be changed via WebSocket message
    current_exercise = ExerciseType.SQUAT
    last_evaluation = None

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        await websocket.close()
        return

    print("=== Gym Form Tracker Server Started ===")
    print(f"Current exercise: {current_exercise.value}")
    print("Send JSON messages to change exercise:")
    print('  {"exercise": "squat"}')
    print('  {"exercise": "bench_press"}')
    print('  {"exercise": "pushup"}')
    print("-" * 50)

    try:
        while True:
            # Check for incoming messages (exercise changes)
            try:
                message = await asyncio.wait_for(websocket.receive_text(), timeout=0.001)
                try:
                    data = json.loads(message)
                    if "exercise" in data:
                        exercise_map = {
                            "squat": ExerciseType.SQUAT,
                            "bench_press": ExerciseType.BENCH_PRESS,
                            "pushup": ExerciseType.PUSHUP
                        }
                        if data["exercise"] in exercise_map:
                            current_exercise = exercise_map[data["exercise"]]
                            detector.reset()
                            last_evaluation = None
                            print(f"\nSwitched to: {current_exercise.value.upper()}")
                except json.JSONDecodeError:
                    pass
            except asyncio.TimeoutError:
                pass
            
            ret, frame = cap.read()
            if not ret:
                break

            # Process Frame with MediaPipe
            timestamp_ms = int((time.time() - start_time) * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = tracker.landmarker.detect_for_video(mp_image, timestamp_ms)
            h, w, _ = frame.shape

            warning_message = None
            
            if result.pose_landmarks:
                # Extract points
                points = extract_points_from_result(result, tracker, h, w)
                
                # Process with pose detector
                timestamp_sec = time.time() - start_time
                evaluation, warning = detector.process_frame(points, current_exercise, timestamp_sec)
                
                warning_message = warning
                
                # If rep completed, update last evaluation
                if evaluation:
                    last_evaluation = evaluation
                    print(f"\n{'='*60}")
                    print(f"Rep {evaluation.rep_number} completed!")
                    print(f"Correct form: {evaluation.is_correct}")
                    print(f"\n{evaluation.feedback_message}")
                    print('='*60)
                    
                    # Audio Feedback Logic 
                    if evaluation.is_correct:
                        # Option A: Simple praise
                        speak(f"Good rep!")
                    else:
                        # Option B: Speak only the PRIMARY error
                        if evaluation.errors:
                            # Reads: "Correction needed. Not deep enough."
                            speak(f"Correction. {evaluation.errors[0]}")
                        else:
                            speak("Check your form.")
                    

                    # Send evaluation to client (as separate message)
                    eval_data = {
                        "type": "evaluation",
                        "rep_number": evaluation.rep_number,
                        "is_correct": evaluation.is_correct,
                        "errors": evaluation.errors,
                        "warnings": evaluation.warnings,
                        "feedback": evaluation.feedback_message
                    }
                    # Send evaluation as JSON with a prefix to distinguish it
                    await websocket.send_text("EVAL:" + json.dumps(eval_data))
                
                # Draw skeleton with feedback
                draw_skeleton_with_feedback(
                    frame, points, warning_message, last_evaluation,
                    current_exercise, detector.rep_count, detector.current_phase
                )
                
            else:
                # No pose detected
                cv2.putText(frame, "No Pose Detected", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Stand in frame - side view", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Controls info
            controls_y = frame.shape[0] - 30
            cv2.putText(frame, f"Current: {current_exercise.value.replace('_', ' ').title()}", 
                        (10, controls_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Encode frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                continue
            
            # Convert to Base64 and send (send raw base64, not wrapped in JSON)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            await websocket.send_text(img_b64)
            
            # Yield control to event loop
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("\nClient disconnected")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        print(f"\nSession complete! Total reps: {detector.rep_count}")


@app.get("/")
async def root():
    return {
        "message": "Gym Form Tracker WebSocket Server",
        "endpoints": {
            "websocket": "/ws",
            "exercises": ["squat", "bench_press", "pushup"]
        },
        "usage": {
            "connect": "ws://localhost:8000/ws",
            "change_exercise": '{"exercise": "squat"}',
            "receive": {
                "frames": "Base64 encoded image strings",
                "evaluations": 'Messages starting with "EVAL:" followed by JSON'
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)