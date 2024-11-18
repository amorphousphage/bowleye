import cv2
import time

# Load the video file
video_path = "pins_new_12.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()
start_time = time.time()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    exit()
frame_number = 1
# Loop through all frames manually
while True:
    # Display the current frame
    #cv2.imshow("Frame Viewer", frame)

    ret, frame = cap.read()
    frame_number += 1
        
    if not ret:
        time_spent = time.time() - start_time
        fps = int(frame_number / time_spent)
        print("End of video reached. FPS: ", fps)
        break

# Release the video capture object and close the window
cap.release()
#cv2.destroyAllWindows()

