# import used packages
import numpy as np
from collections import deque
import os
import cv2
import configparser
import ast
import shutil
import time
from ball_tracker import TrackVideo
from scorer import PinScorer
from signal_router import signal_router
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QObject, Qt, QEventLoop
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage
from threading import Lock, Event

# Generate a class for the recorder to run in a thread
class RecorderWorker(QThread):
    # Define a signal to pass the status of the recorder to the lane tab for the user to see
    recorder_status = pyqtSignal(str)

    # Define signals to tell the pin camera recorder to start and stop buffering frames for the pin video
    start_pins_buffering = pyqtSignal()
    stop_pins_buffering = pyqtSignal()

    def __init__(self, lane_number):
        super().__init__()

        # Obtain lane number for the Recorder Worker
        self.lane_number = lane_number

        # Set the running mode to active
        self.running = True

    # Function to load all necessary settings
    def InitializeSettings(self):

        # Load the settings for the lane
        config = configparser.ConfigParser()
        if os.path.exists(f'settings_lane_{self.lane_number}.cfg'):
            config.read(f'settings_lane_{self.lane_number}.cfg')
        else:
            QMessageBox.critical(None, "No Settings found", "No settings for lane " + str(
                self.lane_number) + " were found. Please go to settings and choose the lane to autogenerate the settings file. Afterwards reboot the program.")
            return False

        self.config = config

        # Read all the settings needed
        self.start_detection_bounds = np.array(ast.literal_eval(config.get('Recorder', 'recorder_start_bounds')),
                                               dtype=np.int32)
        self.end_detection_bounds = np.array(ast.literal_eval(config.get('Recorder', 'recorder_end_bounds')),
                                             dtype=np.int32)
        self.tracking_camera_width = config.getint('Recorder', 'tracking_camera_x_resolution')
        self.tracking_camera_height = config.getint('Recorder', 'tracking_camera_y_resolution')
        self.pins_camera_width = config.getint('Recorder', 'pins_camera_x_resolution')
        self.pins_camera_height = config.getint('Recorder', 'pins_camera_y_resolution')

        self.pins_flipped = config.get('Recorder', 'pins_flipped')

        self.reference_frame_distance = config.getint('Recorder', 'reference_frame_distance')
        self.frames_before_detection = round(config.getfloat('Recorder','time_before_detection') * 25)
        self.frames_after_shot = round(config.getfloat('Recorder', 'time_after_shot') * 25)
        self.frames_after_shot_restore = self.frames_after_shot
        self.export_video_buffer_length = config.getint('Recorder', 'export_video_buffer')
        self.pins_video_frame_buffer = deque(maxlen=60 * self.export_video_buffer_length)
        self.preshot_video_frame_buffer = deque(maxlen= self.frames_before_detection)

        self.ksize = self.config.getint('Ball Detection', 'blurred_kernel')
        self.sigma = self.config.getint('Ball Detection', 'blurred_sigma')
        self.binary_threshold = self.config.getint('Ball Detection', 'binary_threshold')
        self.binary_maximum = self.config.getint('Ball Detection', 'binary_max')
        self.min_radius = self.config.getint('Ball Detection', 'min_radius')
        self.max_radius = self.config.getint('Ball Detection', 'max_radius')
        self.detection_bound_y_coordinate_near_foulline = max(
            point[1] for detection_bound in self.start_detection_bounds for point in detection_bound)
        self.y_coordinate_threshold = self.config.getint('Ball Detection', 'max_vertical_pixel_difference')
        self.x_coordinate_threshold = self.config.getint('Ball Detection', 'max_horizontal_pixel_difference')
        self.detection_threshold = self.config.getint('Recorder', 'detection_threshold')

        self.show_debugging_image = config.get('Ball Detection', 'show_debugging_image')
        self.debugging_image_type = config.get('Ball Detection', 'debugging_image_type')
        
        self.detection_bounds = np.array(ast.literal_eval(config.get('Lane Setup', 'detection_bounds')), dtype=np.int32)
        
        self.min_x_videoexport = self.detection_bounds[0][3][0] - config.getint('Video Export', 'margins_video_export')
        self.max_x_videoexport = self.detection_bounds[0][2][0] + config.getint('Video Export', 'margins_video_export')
        
        self.time_pin_reading_after_sweeper = config.getint('Pin Scorer', 'time_pin_reading_after_sweeper')

        return True

    # Function to initialize the capture from both cameras
    def InitializeCameras(self):
        # Initialize Tracking Camera with defined properties
        self.cap = cv2.VideoCapture(self.config.get('Recorder', 'tracking_camera_path'))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, 1000) # Set FPS to 1000 to make camera achieve the highest possible FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.tracking_camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.tracking_camera_height)

        # Initialize Pins Camera with defined properties
        self.cap_pins = cv2.VideoCapture(self.config.get('Recorder', 'pins_camera_path'))
        self.cap_pins.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap_pins.set(cv2.CAP_PROP_FPS, 1000) # Set FPS to 1000 to make camera achieve the highest possible FPS
        self.cap_pins.set(cv2.CAP_PROP_FRAME_WIDTH, self.pins_camera_width)
        self.cap_pins.set(cv2.CAP_PROP_FRAME_HEIGHT, self.pins_camera_height)
        
        # Set events to signal once the cameras have shut down when stopping the recorder to trigger release of the captures
        self.tracking_frame_stopped_event = Event()
        self.pins_frame_stopped_event = Event()

        # Create worker threads for both cameras
        self.tracking_camera_worker = FrameCaptureWorker(self.cap, self.tracking_frame_stopped_event, self.lane_number, is_pins_camera=False, video_flipped="No")
        self.pins_camera_worker = FrameCaptureWorker(self.cap_pins, self.pins_frame_stopped_event, self.lane_number, is_pins_camera=True, video_flipped=self.pins_flipped)
        
        # Connect the signals to the appropriate slots
        self.tracking_camera_worker.tracking_frame_captured.connect(self.ProcessTrackingCameraFrame, type=Qt.QueuedConnection)
        self.pins_camera_worker.pins_frame_captured.connect(self.ProcessPinsCameraFrame, type=Qt.QueuedConnection)
        
        self.start_pins_buffering.connect(self.pins_camera_worker.StartBuffering)
        self.stop_pins_buffering.connect(self.pins_camera_worker.StopBuffering)
        
        self.tracking_camera_worker.camera_error_signal.connect(self.ShowCameraError)
        self.pins_camera_worker.camera_error_signal.connect(self.ShowCameraError)
        
        return True

    # Function to store frames from the tracking camera
    @pyqtSlot(object)
    def ProcessTrackingCameraFrame(self, frame):
        with self.tracking_frame_lock:
            self.tracking_camera_frame = frame

        # Set the event to signalize the first frame has been captured since the camera was (re)started
        if not self.tracking_frame_ready_event.is_set():
            self.tracking_frame_ready_event.set()

    # Function to process frames from the pins camera
    @pyqtSlot(object)
    def ProcessPinsCameraFrame(self, frame_pins):
        # Flip the frame if required, immediately after receiving it
        if self.pins_flipped == "Yes":
            frame_pins = cv2.flip(frame_pins, -1)
        
         # Update pins_camera_frame with the frame
        with self.pins_frame_lock:
            self.pins_camera_frame = frame_pins
            
        # Set the event to signalize the first frame has been captured since the camera was (re)started
        if not self.pins_frame_ready_event.is_set():
            self.pins_frame_ready_event.set()

    # Function to show the Error message if a camera is not accessible
    @pyqtSlot(str, str)
    def ShowCameraError(self, title, error_message):
        QMessageBox.critical(None, title, error_message)

        # Stop Monitoring and exit the recorder if the camera failed
        self.StopMonitoring()

    # Function to generate a difference image and a binary image depending on the need for it
    def RenderDifferenceImage(self, frame, reference_frame, source, mode):
        # Convert frame and reference_frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        reference_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

        # Mask a region of interest to the image if it comes from the ball tracking camera
        if source == "ball tracking":

            # If the difference image is used to trigger the recorder, mask either the start or end region
            if mode == "recording":
                if self.detection_region == "start":
                    roi_frame = cv2.fillPoly(np.zeros_like(gray), self.start_detection_bounds, 255)
                    roi_reference = cv2.fillPoly(np.zeros_like(reference_frame_gray), self.start_detection_bounds, 255)
                elif self.detection_region == "end":
                    roi_frame = cv2.fillPoly(np.zeros_like(gray), self.end_detection_bounds, 255)
                    roi_reference = cv2.fillPoly(np.zeros_like(reference_frame_gray), self.end_detection_bounds, 255)

            # If the difference image is used for final ball tracking or debugging, mask the detection_bounds of the lane
            elif mode == "tracking" or mode == "debugging":
                roi_frame = cv2.fillPoly(np.zeros_like(gray), self.detection_bounds, 255)
                roi_reference = cv2.fillPoly(np.zeros_like(reference_frame_gray), self.detection_bounds, 255)

            # Mask the grayscale frame with the ROI
            roi_gray = cv2.bitwise_and(gray, gray, mask=roi_frame)

            # Mask the grayed reference with the ROI
            reference_roi_gray = cv2.bitwise_and(reference_frame_gray, reference_frame_gray, mask=roi_reference)

            # Apply a blur to the grayscale frame to reduce noise
            blurred = cv2.GaussianBlur(roi_gray, (self.ksize, self.ksize), self.sigma)

            # Add Blur to the Reference ROI image to reduce noise
            blurred_reference = cv2.GaussianBlur(reference_roi_gray, (self.ksize, self.ksize), self.sigma)

        # Do not mask a region of interest to the image if it comes from the pins camera
        elif source == "pins":
            # Apply a blur to the grayscale frame to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.ksize, self.ksize), self.sigma)

            # Add Blur to the Reference ROI image to reduce noise
            blurred_reference = cv2.GaussianBlur(reference_frame_gray, (self.ksize, self.ksize), self.sigma)

        # Compute the absolute difference between the current frame and the reference frame within the ROI
        diff_frame = cv2.absdiff(blurred, blurred_reference)
        
        # Apply binary thresholding to segment the differences (ball or pins or sweeper)
        _, binary = cv2.threshold(diff_frame, self.binary_threshold, self.binary_maximum, cv2.THRESH_BINARY)

        # If this function was called for tracking instead of debugging, just return the binary image
        if mode == "tracking" or mode == "recording":
            return binary

        # generate the QImage for the debugging image if it should be displayed live, regardless of source of the image
        elif mode == "debugging" and self.show_debugging_image == "Yes" and self.debugging_image_type == "Binary":
            # Crop the image if it comes from the ball tracking camera, but not from the pins camera
            if source == "ball tracking":
                binary = np.ascontiguousarray(binary[:, self.min_x_videoexport:self.max_x_videoexport])
            elif source == "pins":
                binary = np.ascontiguousarray(binary)

            height, width = binary.shape
            q_image = QImage(binary.data, width, height, width, QImage.Format_Grayscale8)

            # Emit the QImage with the correct signal
            if source == "ball tracking":
                signal_router.debugging_image.emit(q_image)
            elif source == "pins":
                signal_router.debugging_image_pins.emit(q_image)

        elif mode == "debugging" and self.show_debugging_image == "Yes" and self.debugging_image_type == "Difference only":
            # Crop the image if it comes from the ball tracking camera, but not from the pins camera
            if source == "ball tracking":
                diff_frame = np.ascontiguousarray(diff_frame[:, self.min_x_videoexport:self.max_x_videoexport])
            elif source == "pins":
                diff_frame = np.ascontiguousarray(diff_frame)

            height, width = diff_frame.shape
            q_image = QImage(diff_frame.data, width, height, width, QImage.Format_Grayscale8)

            # Emit the QImage with the correct signal
            if source == "ball tracking":
                signal_router.debugging_image.emit(q_image)
            elif source == "pins":
                signal_router.debugging_image_pins.emit(q_image)

    # Function to process a finalized tracked frame sent back by the ball tracker
    @pyqtSlot(object)
    def ReceiveTrackedFrame(self, frame):
        # Append the tracked frame to the buffer for later export
        self.tracking_buffer.append(frame)

    # Function to detected circles for the recorder
    def DetectCircles(self, binary, centerlist):

        # Try to find circles in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # For each identified circle, if it fulfills the criteria of belonging to the ball, add it to the centerlist
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center_bottom = (int(x), int(y + radius))

            if self.min_radius < radius < self.max_radius:
                centerlist.append(center_bottom)
        return centerlist

    # Function to detect the sweeper in the pins camera
    def DetectSweeper(self, binary):

        # Set the sweeper count from last frame for comparison
        self.previous_sweeper_count = self.sweeper_count

        # Define the starting row in pixels (50 pixels away from the top)
        row_index = 50

        # Loop through all pixel rows with a space of 10 pixels and check whether all pixels of that row are white (indicating a picture change done by the sweeper)
        while row_index < binary.shape[0] and self.sweeper_detected == False:
            if np.all(binary[row_index, :] == self.binary_maximum):
                self.sweeper_count += 1
                break
            row_index += 10

        # Include mechanism to clear wrong single sweeper detections after a while
        if self.sweeper_count == self.previous_sweeper_count:
            self.frames_without_continuous_sweeper_detection += 1

        elif self.sweeper_count > self.previous_sweeper_count:
            self.frames_without_continuous_sweeper_detection = 0

        if self.frames_without_continuous_sweeper_detection > 4:
            self.sweeper_count = 0

        # Check if enough concurrent sweeper counts have occured to define it as successfully detected
        if self.sweeper_count >= 3:
            self.sweeper_detected = True

            # For debugging, print if a sweeper was detected
            print("Sweeper down detected")

    # Function to wait for the pin video export completion from the FrameCaptureWorker before starting to monitor for new shots
    def WaitForPinVideoExportCompletion(self):
        # Create a temporary event loop
        self.pin_export_wait_loop = QEventLoop()
        # Connect the signal to the event loop's quit method
        signal_router.pins_video_available.connect(self.pin_export_wait_loop.quit)
        # Start the event loop, which will block until pins_video_available is emitted
        self.pin_export_wait_loop.exec_()
        self.pin_export_wait_loop = None
        
    # Function to execute once the recorder is called
    def run(self):
        
        # Obtain the settings
        if not self.InitializeSettings():
            signal_router.finished.emit()
            return

        # Initialize the cameras
        if not self.InitializeCameras():
            signal_router.finished.emit()
            return

        # Set the variables for controlling the recorder loop
        centerlist = []
        self.frames_without_detection = 0
        self.detection_counter = 0
        self.detection_region = "start"
        cut_trigger = "inactive"
        self.lock_preshot_buffer = False
        self.started_pins_buffering = False
        self.start_tracking_timer = False
        
        # Define the variables holding the frames read from the cameras
        self.tracking_camera_frame = None
        self.pins_camera_frame = None
        
        # Define the variable holding all tracked frames
        self.tracking_buffer = []
        
        # Define the variable to hold the reference frame of the tracking camera
        self.reference_frame = None

        # Define the variables used for sweeper detection
        self.frames_without_continuous_sweeper_detection = 0
        self.frames_after_sweeper_detection = 0
        self.sweeper_detected = False
        self.sweeper_count = 0

        # Define the variables to hold the pins reference and reading frame for score reading
        self.pin_scorer_ref_frame = None
        self.pin_scorer_reading_frame = None

        # Set the locks for the tracking and pin camera such that not more than one thread can access the shared variable at once
        self.tracking_frame_lock = Lock()
        self.pins_frame_lock = Lock()
        
        # Set events to signal once the cameras have been (re)started and a frame has been read
        self.tracking_frame_ready_event = Event()
        self.pins_frame_ready_event = Event()
        
        # Start the camera workers
        self.tracking_camera_worker.start()
        self.pins_camera_worker.start()

        # Start the recording for the tracking camera
        self.tracking_camera_worker.recording_active_event.set()
        
        # Wait for the tracking camera to be ready and have its first frame captured
        self.tracking_frame_ready_event.wait()
        
        # Initialize the BallTracker with a captured frame
        with self.tracking_frame_lock:
            frame = self.tracking_camera_frame
        self.ball_tracker = TrackVideo(frame, self.lane_number, frame)
        self.ball_tracker.InitializeTracker()
        
        # Connect signals receiving tracked frames from the ball tracker
        signal_router.finalized_tracked_frame.connect(self.ReceiveTrackedFrame)

        # Emit a signal to show that the recorder is idle and ready to record
        self.recorder_status.emit("idle")

        # Define a loop that runs, while the recorder is active
        while self.running:

            # Check if the preshot buffer is not locked (ball was not yet detected and the preshot frame buffer should still be overwritten), then append the frame
            if not self.lock_preshot_buffer:
                with self.tracking_frame_lock:
                    frame = self.tracking_camera_frame
                self.preshot_video_frame_buffer.append(frame)
            
            # Generate and show the debugging image if enabled and if a reference_frame has been set
            if self.show_debugging_image == "Yes" and self.reference_frame is not None:
                self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "debugging")
               
            # Check if we are past the ball entering the pins (cut_trigger active)
            if cut_trigger == "inactive":

                # Obtain the reference frame in the user defined distance from the current frame if possible.
                if len(self.preshot_video_frame_buffer) < self.reference_frame_distance:
                    self.reference_frame = self.preshot_video_frame_buffer[0]
                else:
                    self.reference_frame = self.preshot_video_frame_buffer[-self.reference_frame_distance]

                # Set the amount of circles detected from previous detection round
                self.centerlist_length = len(centerlist)

                # Generate the binary difference image from the current ball tracking frame to it's reference (called in recording mode so the masking is done to either start or end detection bounds)
                with self.tracking_frame_lock:
                    frame = self.tracking_camera_frame
                binary_image = self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "recording")
                
                # Detect circles in the current frame
                centerlist = self.DetectCircles(binary_image, centerlist)

                # If additional circles compared to last round were detected, increase the detection counter. If not, increase the counter for frames without detection
                if len(centerlist) > self.centerlist_length:
                    self.detection_counter += 1
                    self.frames_without_detection = 0
                else:
                    self.frames_without_detection += 1

                # Reset the Detection Counter if enough frames without detection have passed (likely a false circle was detected in a previous timeframe)
                if self.frames_without_detection > 20:
                    self.detection_counter = 0

                # If enough centers at the start of the lane have been registered, switch the detection to the end, reset the counters and start recording
                elif self.detection_counter >= self.detection_threshold and self.detection_region == "start":
                    self.detection_region = "end"
                    self.detection_counter = 0
                    self.frames_without_detection = 0
                    centerlist=[]

                    # Lock the pre-shot buffer, so it is not overwritten anymore, because the shot has started
                    self.lock_preshot_buffer = True

                    # Emit a signal to show that the recorder is now recording
                    self.recorder_status.emit("recording")
                    
                # If not enough circles have been detected at the end of the lane, keep recording
                elif self.detection_counter < self.detection_threshold and self.detection_region == "end":
                    # Start the timer for ball tracking
                    if not self.start_tracking_timer:
                        self.start_tracking_timer = True
                        start_time_ball_tracking = time.time()

                    # Generate the difference image to send to the tracker
                    with self.tracking_frame_lock:
                        frame = self.tracking_camera_frame
                    binary_image = self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "tracking")

                    # Send the current frame to the ball tracker for tracking
                    self.ball_tracker.TrackFrame(binary_image, frame)

                # If enough circles at the end of the lane have been detected, activate the cut trigger (run out period to record the pins camera) and keep recording
                elif self.detection_counter >= self.detection_threshold and self.detection_region == "end":
                    cut_trigger = "active"
                    self.frames_after_shot -= 1
                    self.frame_after_sweeper_detection = 0

                    # Generate the difference image to send to the tracker
                    with self.tracking_frame_lock:
                        frame = self.tracking_camera_frame
                    binary_image = self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "tracking")

                    # Send the current frame to the ball tracker for tracking
                    self.ball_tracker.TrackFrame(binary_image, frame)

                    # Start the pins camera if it is not running
                    if not self.pins_camera_worker.recording_active_event.is_set():
                        self.pins_camera_worker.recording_active_event.set()
                        # Wait for the first pins camera image
                        self.pins_frame_ready_event.wait()

                    # Start the pins buffering if not started yet
                    if not self.started_pins_buffering:
                        self.start_pins_buffering.emit()
                        self.started_pins_buffering = True

            # If the cut trigger is active and there are still frames to be recorded, record them as well as the ones for the pin camera
            elif cut_trigger == "active" and self.frames_after_shot > -1:

                # Generate the difference image to send to the tracker
                with self.tracking_frame_lock:
                    frame = self.tracking_camera_frame
                binary_image = self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "tracking")

                # Send the current frame to the ball tracker for tracking
                self.ball_tracker.TrackFrame(binary_image, frame)
                
                # If not set, set the pins reference frame
                if self.pin_scorer_ref_frame is None:
                    with self.pins_frame_lock:
                        self.pin_scorer_ref_frame = self.pins_camera_frame
                
                # Obtain the current image from the pins camera
                with self.pins_frame_lock:
                    pins_frame = self.pins_camera_frame

                # Generate and show the debugging image if enabled and if a reference_frame has been set
                if self.show_debugging_image == "Yes" and self.pin_scorer_ref_frame is not None:
                    self.RenderDifferenceImage(pins_frame, self.pin_scorer_ref_frame, "pins", "debugging")

                # Increase the frame count after sweeper detection by one
                if self.sweeper_detected == True:
                    self.frame_after_sweeper_detection += 1

                # Generate binary difference image of the pins frame for sweeper detection
                if self.sweeper_detected == False and self.pin_scorer_ref_frame is not None:
                    binary_image_pins = self.RenderDifferenceImage(pins_frame, self.pin_scorer_ref_frame,"pins", "tracking")

                    # Try to detect the sweeper in the binary image (a row of all white pixels)
                    self.DetectSweeper(binary_image_pins)

                # If time after sweeper detection has been reached, capture the read frame for the pins scoring
                if self.frame_after_sweeper_detection == self.time_pin_reading_after_sweeper:
                    self.pin_scorer_reading_frame = pins_frame

                # Check if the last frame is reached
                if self.frames_after_shot == 0:

                    # If no sweeper was detected, take the last frame as reading frame for the pin scoring
                    if self.sweeper_detected == False:
                        print("No sweeper was detected during this shot")
                        with self.pins_frame_lock:
                            self.pin_scorer_reading_frame = pins_frame
                    
                    # Stop the timer for the ball tracking
                    if self.start_tracking_timer:
                        self.start_tracking_timer = False
                        stop_time_ball_tracking = time.time()
                        
                        # Calculate FPS of the Ball Tracking Camera
                        self.tracking_fps = int(len(self.tracking_buffer) / (stop_time_ball_tracking - start_time_ball_tracking))
                        print("Ball Tracker FPS: ", self.tracking_fps)

                    # Emit a signal to show that the recorder is now saving the video files
                    self.recorder_status.emit("generating video")

                    # Stop the pins buffering
                    if self.started_pins_buffering:
                        self.stop_pins_buffering.emit()
                        self.started_pins_buffering = False

                    # Pause recording of both cameras
                    if self.tracking_camera_worker.recording_active_event.is_set():
                        self.tracking_camera_worker.recording_active_event.clear()
                    if self.pins_camera_worker.recording_active_event.is_set():
                        self.pins_camera_worker.recording_active_event.clear()

                    # Trigger the Score Reading in a seperate thread to run asynchronously
                    self.scorer_worker = ScorerWorker(self.pin_scorer_ref_frame, self.pin_scorer_reading_frame, ast.literal_eval(self.config.get('Pin Scorer', 'pin_coordinates')))
                    self.scorer_worker.start()

                # Reduce the frames after shot by one
                self.frames_after_shot -= 1

            # If the last frame to be recorded is reached, write the tracking video, then trigger the analysis and reset the variables for the next shot
            elif cut_trigger == "active" and self.frames_after_shot <= -1:
                
                # Write pin score reference frame as pin image to be statically displayed
                cv2.imwrite('videos/pins_new_' + str(self.lane_number) + '.png', self.pin_scorer_ref_frame)
                
                # Initialize the Tracking Video Writer and write all the tracking frames to file
                self.output_path_tracking = os.path.join('videos', f'tracked_new_{self.lane_number}.mp4')
                frame_height, frame_width = self.tracking_buffer[0].shape[:2]
                self.out = cv2.VideoWriter(self.output_path_tracking, cv2.VideoWriter_fourcc(*'mp4v'), self.tracking_fps, (frame_width, frame_height))
                
                for tracking_frame in self.tracking_buffer:
                    self.out.write(tracking_frame)
                 
                # Emit a signal to show that the ball_tracker is now calculating all values of the shot
                self.recorder_status.emit("calculating")
                
                # Perform all calculations and visualizations on the tracked frames
                self.ball_tracker.CalculateAndDraw()

                # Write the last frame a couple of times into the video which includes the visualized calculations
                for _ in range (int(self.tracking_fps * 0.3)):
                    self.out.write(self.tracking_buffer[-1])
                    
                # Release the writer
                self.out.release()

                # Emit a signal to show that the recorder is now resetting itself for the next shot
                self.recorder_status.emit("resetting")

                # Reset all the necessary variables
                centerlist = []
                self.frames_without_detection = 0
                self.detection_region = "start"
                cut_trigger = "inactive"
                self.detection_counter = 0
                self.preshot_video_frame_buffer.clear()
                self.preshot_video_frame_buffer = deque(maxlen=self.frames_before_detection)
                self.frames_after_shot = self.frames_after_shot_restore
                self.lock_preshot_buffer = False
                self.frames_without_continuous_sweeper_detection = 0
                self.frames_after_sweeper_detection = 0
                self.sweeper_detected = False
                self.sweeper_count = 0
                self.pin_scorer_ref_frame = None
                self.pin_scorer_reading_frame = None
                with self.pins_frame_lock:
                    self.pins_camera_frame = None
                #with self.tracking_frame_lock:
                #    self.tracking_camera_frame = None
                frame = None
                pins_frame = None
                self.tracking_buffer = []
                
                # Re-initialize the Ball Tracker
                self.ball_tracker.InitializeTracker()
                
                # Trigger the event loop to wait for the Pin Video Export Completion before starting to monitor for a new shot
                self.recorder_status.emit("waiting for pin export completion")
                
                self.WaitForPinVideoExportCompletion()
                
                # Start the recording of the tracking camera again
                if not self.tracking_camera_worker.recording_active_event.is_set():
                    self.tracking_camera_worker.recording_active_event.set()
                    self.tracking_frame_ready_event.wait()

                # Emit a signal to show that the recorder is idle and ready to record
                self.recorder_status.emit("idle")

        # Stop the camera workers
        self.tracking_camera_worker.stop()
        self.pins_camera_worker.stop()
        
        # Wait for the camera workers to stop
        self.tracking_frame_stopped_event.wait()
        self.pins_frame_stopped_event.wait()
        
        # Release the cameras after the recorder is stopped
        self.cap.release()
        self.cap_pins.release()
        
        # Wait for threads to finish
        self.tracking_camera_worker.wait()
        self.pins_camera_worker.wait()
        
        # Emit a signal to show that the recorder has been turned off
        self.recorder_status.emit("recorder offline")


    # Function to stop the recorder loop
    def StopMonitoring(self):
        self.running = False


# Worker class for frame capturing of both cameras
class FrameCaptureWorker(QThread):

    # Define signals to send the captured frame and the camera error signal to the recorder worker
    tracking_frame_captured = pyqtSignal(object)
    pins_frame_captured = pyqtSignal(object)
    camera_error_signal = pyqtSignal(str, str)

    def __init__(self, camera, stop_event, lane_number, is_pins_camera=False, video_flipped="No"):
        super().__init__()

        # Define used variables locally
        self.camera = camera
        self.stop_event = stop_event
        self.is_pins_camera = is_pins_camera
        self.running = True
        self.buffering = False
        self.buffer_lock = Lock()
        self.recording_active_event = Event()
        self.pins_frame_buffer = []
        self.video_flipped = video_flipped
        self.lane_number = lane_number

    # Function to start writing pins camera frames into a buffer for a shot
    @pyqtSlot()
    def StartBuffering(self):
        # Reset the buffer
        self.pins_frame_buffer = []
        # Start buffering
        self.buffering = True

        # Reset and set the time where buffering started (used for FPS calculation)
        self.start_time_buffer = None
        self.start_time_buffer = time.time()

    # Function to stop writing pins camera frames into a buffer for a shot, calculate the achieved FPS and send the buffer for processing
    @pyqtSlot()
    def StopBuffering(self):
        with self.buffer_lock:
            # Stop buffering
            self.buffering = False

            # Calculate the time spent buffering and from it calculate the FPS
            fps_pins = int(len(self.pins_frame_buffer) / (time.time() - self.start_time_buffer))

        print("Pin Camera FPS: ", fps_pins)

        # Define a thread for asynchronous pin video exporting
        self.pins_video_export_thread = PinsVideoExportWorker(fps_pins, self.pins_frame_buffer, self.video_flipped, os.path.join('videos', f'pins_new_{self.lane_number}.mp4'))

        # Connect the finishing of the thread to a function resetting the pin buffer and sending a completion signal to the main recorder thread
        self.pins_video_export_thread.finished.connect(self.PinsVideoExportComplete)

        # Start the thread
        self.pins_video_export_thread.start()

    def PinsVideoExportComplete(self):

        # Emit signal when completed
        signal_router.pins_video_available.emit()


    def run(self):
        # Attempt to open the camera, retry if not immediately available
        retries = 3
        while retries > 0 and not self.camera.isOpened():
            time.sleep(1)
            retries -= 1

        # If camera could not be opened after retries, show an error
        if not self.camera.isOpened():
            if self.is_pins_camera:
                self.camera_error_signal.emit("Camera not readable", "Pins Camera for this lane could not be accessed after multiple attempts. Please check the camera connection.")
            else:
                self.camera_error_signal.emit("Camera not readable", "Tracking Camera for this lane could not be accessed after multiple attempts. Please check the camera connection.")
            
            self.running = False
            return
            
        # Proceed to capture frames once the camera is opened successfully
        while self.running:
            # If the Recording Active event is not set, wait for it (for the camera to start) and flush the first couple of frames to ensure current frames are being sent
            if not self.recording_active_event.is_set():                
                # Wait if the recording for the camera is not activated until the event to start recording is set
                self.recording_active_event.wait()
 
                # Flush the first couple of frames after (re-starting the camera)
                for _ in range(5):
                    ret, frame = self.camera.read()
                    if not ret:
                        if self.is_pins_camera:
                            self.camera_error_signal.emit("Camera not readable", "Pins Camera for this lane could not be accessed during capture. Please ensure the camera is working.")
                        else:
                            self.camera_error_signal.emit("Camera not readable", "Tracking Camera for this lane could not be accessed during capture. Please ensure the camera is working.")
                        break
               
            
            # Read the frame from the camera
            ret, frame = self.camera.read()
            if not ret:
                if self.is_pins_camera:
                    self.camera_error_signal.emit("Camera not readable", "Pins Camera for this lane could not be accessed during capture. Please ensure the camera is working.")
                else:
                    self.camera_error_signal.emit("Camera not readable", "Tracking Camera for this lane could not be accessed during capture. Please ensure the camera is working.")
                break
            elif frame is None or frame.size == 0:
                print("Frame is empty for ", self.camera)

            # Only emit (and buffer if applicable) the frame if it is not empty
            else:
                # Emit the frame based on whether it's a tracking or pins camera frame
                if self.is_pins_camera:
                    self.pins_frame_captured.emit(frame)

                    # If the pin camera buffer is enabled, write the frame to the pin buffer
                    if self.buffering:
                        self.pins_frame_buffer.append(frame)

                else:
                    self.tracking_frame_captured.emit(frame)

        # Set the event to signal successful stopping of the camera readings (only when the recorder is shut down)
        self.stop_event.set()

    # Function to stop the camera recorder
    def stop(self):
        self.running = False

        # Setting the recording to active to avoid waiting indefinetly so that the while loop can finish and the run function can conclude
        if not self.recording_active_event.is_set():
            self.recording_active_event.set()

# Generate a Thread to asynchronously export the Pins Video Buffer
class PinsVideoExportWorker(QThread):
    def __init__(self, fps_pins, pins_buffer, video_flipped, output_path):
        super().__init__()
        # Define the variables used
        self.fps_pins = fps_pins
        self.pins_buffer = pin_buffer
        self.video_flipped = video_flipped
        self.output_path = output_path

# Function to perform the export
    def run(self):

        # Obtain the height and width of the frames to export
        pins_frame_height, pins_frame_width = self.pins_buffer[0].shape[:2]

        # Initialize a video writer
        out_pins = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps_pins, (pins_frame_width, pins_frame_height))
        
        # Export all frames
        for frame in self.pins_buffer:
            if self.video_flipped == "Yes":
                frame = cv2.flip(frame, -1)
            out_pins.write(frame)

        # Release the video writer
        out_pins.release()
        self.pins_buffer = None

# Create a class to read the pin score in a thread and pass the score to the lane_tab.py for display
class ScorerWorker(QThread):
    def __init__(self, ref_frame, reading_frame, pin_coordinates, parent=None):
        super().__init__(parent)

        # Define the used variables
        self.ref_frame = ref_frame
        self.reading_frame = reading_frame
        self.pin_coordinates = pin_coordinates

    # Function to read the score (pins left standing)
    def run(self):
        # Initialize the PinScorer and process the frames
        scorer = PinScorer(self.ref_frame, self.pin_coordinates)
        standing_pins = scorer.PinsStillStanding(self.reading_frame)

        # Emit the result
        signal_router.pins_standing_signal.emit(standing_pins)


