import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open video file (change filename if needed)
video_path = "/home/tonyzalez/Projects_learning/projects/squat_RIR/DSCF0415.AVI"
cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video (optional)
output_path = "squat_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB (for MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, _ = frame.shape
            # Normalize mediapipes fractional coordinates realtive to width and height(0-1) into pixel coordinates for opencv
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Extract key joints (hip & knee) for squat analysis
        hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        
        # Convert normalized coordinates to pixel values
        hip_y = int(hip.y * h)
        knee_y = int(knee.y * h)

        # Display depth information
        cv2.putText(frame, f"Hip Y: {hip_y}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Knee Y: {knee_y}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the video frame
    cv2.imshow("Squat Analysis", frame)
    
    # Save the processed video (optional)
    out.write(frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
