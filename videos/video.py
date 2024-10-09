import cv2

# Load the video file
video_path = "pins_new_12.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Error: Could not read video file.")
    exit()

# Loop through all frames manually
while True:
    # Display the current frame
    cv2.imshow("Frame Viewer", frame)

    # Wait for key event
    key = cv2.waitKey(0)

    # Check the pressed key
    if key == ord('n'):  # Press 'n' to go to the next frame
        ret, frame = cap.read()
        if not ret:
            print("End of video reached.")
            break
    elif key == ord('p'):  # Press 'p' to go to the previous frame (not supported in OpenCV, just an example)
        print("Previous frame functionality is not supported in OpenCV directly.")
    elif key == ord('q'):  # Press 'q' to quit
        print("Exiting...")
        break

# Release the video capture object and close the window
cap.release()
cv2.destroyAllWindows()
