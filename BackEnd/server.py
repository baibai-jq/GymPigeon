# from fastapi import FastAPI, WebSocket
# import asyncio

# app = FastAPI()

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         while True:
#             # Receive message from React client
#             data = await websocket.receive_text()
#             # Send response back to React client
#             await websocket.send_text(f"Server received: {data}")
#     except Exception as e:
#         print(f"Connection closed: {e}")

import cv2
import mediapipe as mp
import time
import base64
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from landmarks import PoseTracker, print_visible_landmarks
import numpy as np

app = FastAPI()

# --- Helper Function (Copied from BackEnd/main.py) ---
# def draw_skeleton(frame, key, side_color=(255, 0, 0)):
#     h, w, _ = frame.shape
#     pts = {}
#     for name, p in key.__dict__.items():
#         if p.visibility >= 0.5:
#             cx, cy = int(p.x * w), int(p.y * h)
#             pts[name] = (cx, cy)
#             cv2.circle(frame, (cx, cy), 6, side_color, -1)

#     connections = [
#         ("shoulder", "elbow"), ("elbow", "wrist"),
#         ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
#     ]

#     for a, b in connections:
#         if a in pts and b in pts:
#             cv2.line(frame, pts[a], pts[b], side_color, 3)

def calculate_angle(a, b, c):
    """
    Calculates the angle at point B given points A, B, and C.
    Args:
        a, b, c: tuples of (x, y) coordinates
    Returns:
        angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
    
    return angle

def draw_skeleton(frame, result, tracker):
    """Draw complete skeleton with angle measurements"""
    if not result.pose_landmarks:
        return
    
    h, w, _ = frame.shape
    lm = result.pose_landmarks[0]
    
    # Extract all landmark points
    points = {}
    
    # Right side landmarks
    for name, idx in tracker.RIGHT.items():
        if lm[idx].visibility >= 0.5:
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            points[f"right_{name}"] = (cx, cy)
            cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
    
    # Left side landmarks
    for name, idx in tracker.LEFT.items():
        if lm[idx].visibility >= 0.5:
            cx, cy = int(lm[idx].x * w), int(lm[idx].y * h)
            points[f"left_{name}"] = (cx, cy)
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
    
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
            if "left" in point_a and "left" in point_b:
                color = (0, 0, 255)
            elif "right" in point_a and "right" in point_b:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            
            cv2.line(frame, points[point_a], points[point_b], color, 3)
    
    # Calculate and display angles
    angles = {}
    
    # Right arm angle (shoulder-elbow-wrist)
    if all(k in points for k in ["right_shoulder", "right_elbow", "right_wrist"]):
        angle = calculate_angle(
            points["right_shoulder"],
            points["right_elbow"],
            points["right_wrist"]
        )
        angles["right_elbow"] = angle
        # Display angle near the elbow
        pos = points["right_elbow"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Left arm angle (shoulder-elbow-wrist)
    if all(k in points for k in ["left_shoulder", "left_elbow", "left_wrist"]):
        angle = calculate_angle(
            points["left_shoulder"],
            points["left_elbow"],
            points["left_wrist"]
        )
        angles["left_elbow"] = angle
        pos = points["left_elbow"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Right shoulder angle (elbow-shoulder-hip)
    if all(k in points for k in ["right_elbow", "right_shoulder", "right_hip"]):
        angle = calculate_angle(
            points["right_elbow"],
            points["right_shoulder"],
            points["right_hip"]
        )
        angles["right_shoulder"] = angle
        pos = points["right_shoulder"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Left shoulder angle (elbow-shoulder-hip)
    if all(k in points for k in ["left_elbow", "left_shoulder", "left_hip"]):
        angle = calculate_angle(
            points["left_elbow"],
            points["left_shoulder"],
            points["left_hip"]
        )
        angles["left_shoulder"] = angle
        pos = points["left_shoulder"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Right knee angle (hip-knee-ankle)
    if all(k in points for k in ["right_hip", "right_knee", "right_ankle"]):
        angle = calculate_angle(
            points["right_hip"],
            points["right_knee"],
            points["right_ankle"]
        )
        angles["right_knee"] = angle
        pos = points["right_knee"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Left knee angle (hip-knee-ankle)
    if all(k in points for k in ["left_hip", "left_knee", "left_ankle"]):
        angle = calculate_angle(
            points["left_hip"],
            points["left_knee"],
            points["left_ankle"]
        )
        angles["left_knee"] = angle
        pos = points["left_knee"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Right hip angle (shoulder-hip-knee)
    if all(k in points for k in ["right_shoulder", "right_hip", "right_knee"]):
        angle = calculate_angle(
            points["right_shoulder"],
            points["right_hip"],
            points["right_knee"]
        )
        angles["right_hip"] = angle
        pos = points["right_hip"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Left hip angle (shoulder-hip-knee)
    if all(k in points for k in ["left_shoulder", "left_hip", "left_knee"]):
        angle = calculate_angle(
            points["left_shoulder"],
            points["left_hip"],
            points["left_knee"]
        )
        angles["left_hip"] = angle
        pos = points["left_hip"]
        cv2.putText(frame, f"{int(angle)}°", (pos[0] + 10, pos[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return angles

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize Camera and Tracker
    cap = cv2.VideoCapture(1)
    tracker = PoseTracker()
    start_time = time.time()

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        await websocket.close()
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Process Frame (MediaPipe Logic)
            timestamp_ms = int((time.time() - start_time) * 1000)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            result = tracker.landmarker.detect_for_video(mp_image, timestamp_ms)

            # 2. Draw Skeleton on the original BGR frame
            if result.pose_landmarks:
                # right_key = tracker.extract_key_landmarks(result, side="right")
                # if right_key:
                #     draw_skeleton(frame, right_key, side_color=(255, 0, 0))
                
                # left_key = tracker.extract_key_landmarks(result, side="left")
                # if left_key:
                #     draw_skeleton(frame, left_key, side_color=(0, 0, 255))
                angles = draw_skeleton(frame, result, tracker)


                if angles:
                    print("\n" + "="*50)
                    for joint, angle in angles.items():
                        print(f"{joint.replace('', ' ').title()}: {angle:.1f}°")
                else:
                    cv2.putText(frame, "No Pose Detected", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


            # 3. Encode frame to JPEG
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                continue
            
            # 4. Convert to Base64 string
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # 5. Send to React
            await websocket.send_text(img_b64)
            
            # Yield control to the event loop to ensure stability
            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()