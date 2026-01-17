import cv2
import mediapipe as mp
import time
from landmarks import PoseTracker, print_visible_landmarks

def draw_skeleton(frame, key, side_color=(255, 0, 0)):
    h, w, _ = frame.shape
    pts = {}
    for name, p in key.__dict__.items():
        # Check if the AI is confident enough to show the point
        if p.visibility >= 0.5:
            cx, cy = int(p.x * w), int(p.y * h)
            pts[name] = (cx, cy)
            cv2.circle(frame, (cx, cy), 6, side_color, -1)

    connections = [
        ("shoulder", "elbow"), ("elbow", "wrist"),
        ("shoulder", "hip"), ("hip", "knee"), ("knee", "ankle"),
    ]

    for a, b in connections:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], side_color, 3)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    tracker = PoseTracker()
    start_time = time.time()

    print("System started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate timestamp for MediaPipe VIDEO mode
        timestamp_ms = int((time.time() - start_time) * 1000)

        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Run Pose Detection
        result = tracker.landmarker.detect_for_video(mp_image, timestamp_ms)

        # --- DATA EXTRACTION & PRINTING ---
        if result.pose_landmarks:
            # Handle RIGHT side
            right_key = tracker.extract_key_landmarks(result, side="right")
            if right_key:
                # This prints the X, Y, and Visibility to your terminal
                print_visible_landmarks(right_key, side="Right")
                draw_skeleton(frame, right_key, side_color=(255, 0, 0))

            # Handle LEFT side
            left_key = tracker.extract_key_landmarks(result, side="left")
            if left_key:
                # This prints the X, Y, and Visibility to your terminal
                print_visible_landmarks(left_key, side="Left")
                draw_skeleton(frame, left_key, side_color=(0, 0, 255))
        else:
            cv2.putText(frame, "No Pose Detected", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the standard, full-screen frame
        cv2.imshow("Pose Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()