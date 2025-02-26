import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Open video file (change filename if needed)
video_path = "squat.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties (width, height, FPS)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Save output video (optional)
output_path = "squat_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

rep_count = 0
squat_down = False
initial_velocity = None
prev_hip_y = None
prev_time = None
velocity_list = []
velocity = 0
rep_time = 0
prev_rep_time = 0
best_rep_time = 0
nearing_failure = False

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
            # cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # Extract key joints (hip & knee) for squat analysis
        hip = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
        knee = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]
        
        # Convert normalized coordinates to pixel values
        hip_y = int(hip.y * h)
        knee_y = int(knee.y * h)

        cv2.circle(frame, (int(hip.x * w), int(hip.y * h)), 5, (0, 0, 255), -1)
        cv2.circle(frame, (int(knee.x * w), int(knee.y * h)), 5, (255, 0, 0), -1)

        
         # **Step 1: Calculate Rep Velocity**
        current_time = time.time()  # Get current timestamp

        if prev_hip_y is not None and prev_time is not None:
            time_elapsed = current_time - prev_time
            rep_time += time_elapsed
            velocity = (prev_hip_y - hip_y) / time_elapsed  # Pixels per second
            
            # Store the first velocity as baseline
            if initial_velocity is None:
                initial_velocity = velocity
            
            # Track velocity drop (for RIR estimation)
            velocity_list.append(velocity)

       # **Step 2: Detect Squat Phases (Up/Down)**
            if hip_y > knee_y:  # Going down
                squat_down = True
            elif squat_down and hip_y < knee_y:  # Going up (rep completed)
                rep_count += 1
                squat_down = False
                prev_rep_time = rep_time   #track previous rep time and reset rep time
                rep_time = 0;
                if rep_time > best_rep_time: #best rep time set as basiline
                    best_rep_time = prev_rep_time;
        
        #check if nearing failure based on drop in rep speed
        if prev_rep_time/best_rep_time < 0.7: #30% drop
            nearing_failure: True 
            
        # Update previous values for next frame
        prev_hip_y = hip_y
        prev_time = current_time

        # Display depth information
        cv2.putText(frame, f"Hip Y: {hip_y}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Knee Y: {knee_y}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Reps: {rep_count}", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # cv2.putText(frame, f"Velocity: {velocity:.2f} px/s", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Rep Time: {prev_rep_time}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Nearing Failure: {nearing_failure}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

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
