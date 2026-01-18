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
import requests
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from text_to_speech import speak

load_dotenv()
EVENLAB_API_KEY = os.getenv("EVENLAB_API_KEY")

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

def generate_feedback_data(points, warning_message, evaluation, exercise_type, rep_count, current_phase):
    """Generate feedback data to send to frontend instead of drawing on frame"""
    feedback = {
        "type": "feedback",
        "exercise": exercise_type.value.replace("_", " ").title(),
        "reps": rep_count,
        "phase": current_phase.value.upper(),
        "warning": warning_message,
        "status": None,
        "errors": [],
        "has_evaluation": False
    }
    
    # Add evaluation data if available
    if evaluation:
        feedback["has_evaluation"] = True
        feedback["status"] = "GOOD FORM ✓" if evaluation.is_correct else "CHECK FORM ✗"
        feedback["is_correct"] = evaluation.is_correct
        feedback["errors"] = evaluation.errors[:3]  # Max 3 errors
    
    return feedback

def draw_skeleton_only(frame, points, evaluation):
    """Draw only skeleton on frame without any text"""
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

def evenlab_tts_short_message(message: str):

    audio_generator = client.text_to_speech.convert(
        text=message,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    try:
        audio_bytes = b"".join(audio_generator)
        filename = "temo_output.mp3"
        with open(filename, "wb") as f:
            f.write(audio_bytes)
        os.system(f"afplay {filename}")
    except Exception as e:
        print(f"Evenlab TTS error: {e}")

def get_short_error_message(error: str) -> str:
    """Map detailed error to a short, actionable phrase"""
    error = error.lower()
    if "lean" in error or "forward" in error:
        return "Lean back a bit."
    if "depth" in error or "not deep enough" in error or "lower your hips" in error:
        return "Get a bit lower."
    if "back" in error and "round" in error:
        return "Keep your back straighter."
    if "hips" in error and "sag" in error:
        return "Keep your hips up."
    if "arms" in error and "close" in error:
        return "Widen your grip."
    if "arch" in error:
        return "Reduce your back arch."
    # Default fallback
    return "Check your form."


client = ElevenLabs(api_key=EVENLAB_API_KEY)

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
                        # Option B: Use Evenlab TTS for concise error
                        if evaluation.errors:
                            short_msg = get_short_error_message(evaluation.errors[0])
                            evenlab_tts_short_message(short_msg)
                        else:
                            evenlab_tts_short_message("Check your form.")
                    

                    # Send evaluation to client (as separate message)
                    eval_data = {
                        "type": "evaluation",
                        "rep_number": evaluation.rep_number,
                        "is_correct": evaluation.is_correct,
                        "errors": evaluation.errors,
                        "warnings": evaluation.warnings,
                        "feedback": evaluation.feedback_message
                    }
                    await websocket.send_text("EVAL:" + json.dumps(eval_data))
                    
                    # Clear evaluation after sending so next rep doesn't inherit old errors
                    last_evaluation = None
                
                # Generate feedback data for frontend
                feedback_data = generate_feedback_data(
                    points, warning_message, last_evaluation,
                    current_exercise, detector.rep_count, detector.current_phase
                )
                await websocket.send_text("FEEDBACK:" + json.dumps(feedback_data))
                
                # Draw only skeleton without text
                draw_skeleton_only(frame, points, last_evaluation)
                
            else:
                # No pose detected - no text drawing on frame, frontend will handle
                pass

            # Encode frame to JPEG (clean frame without any text overlays)
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