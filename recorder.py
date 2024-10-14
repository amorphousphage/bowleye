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
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QThread, QObject
from PyQt5.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QLabel, QApplication
from PyQt5.QtGui import QPixmap, QImage
from threading import Lock, Event

# Generate a class for the recorder to run in a thread
class RecorderWorker(QThread):
    # Define a signal to pass the status of the recorder to the lane tab for the user to see
    recorder_status = pyqtSignal(str)
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
        self.frames_before_detection = round(config.getfloat('Recorder','time_before_detection') * self.frame_rate)
        self.frames_after_shot = round(config.getfloat('Recorder', 'time_after_shot') * self.frame_rate)
        self.frames_after_shot_restore = self.frames_after_shot
        self.export_video_buffer_length = config.getint('Recorder', 'export_video_buffer')
        self.pins_video_frame_buffer = deque(maxlen=self.frame_rate_pins * self.export_video_buffer_length)
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
        self.tracking_camera_worker = FrameCaptureWorker(self.cap, self.tracking_frame_stopped_event, is_pins_camera=False)
        self.pins_camera_worker = FrameCaptureWorker(self.cap_pins, self.pins_frame_stopped_event, is_pins_camera=True)
        
        # Connect the signals to the appropriate slots
        self.tracking_camera_worker.tracking_frame_captured.connect(self.ProcessTrackingCameraFrame)
        self.pins_camera_worker.pins_frame_captured.connect(self.ProcessPinsCameraFrame)
        self.pins_camera_worker.buffer_ready.connect(self.ReceivePinsBuffer)
        
        self.start_pins_buffering.connect(self.pins_camera_worker.StartBuffering)
        self.stop_pins_buffering.connect(self.pins_camera_worker.StopBuffering)
        
        return True

    # Function to process tracking camera frames
    def ProcessTrackingCameraFrame(self, frame):
        if self.tracking_camera_frame is None:
            with self.tracking_frame_lock:
                self.tracking_camera_frame = frame
            # Set the event to signalize the first frame has been captured
            self.tracking_frame_ready_event.set()
        else:
            with self.tracking_frame_lock:
                self.tracking_camera_frame = frame

    # Function to process pins camera frames
    def ProcessPinsCameraFrame(self, frame_pins):
        if self.pins_camera_frame is None:
            with self.pins_frame_lock:
                self.pins_camera_frame = frame_pins
            # Set the event to signalize the first frame has been captured
            self.pins_frame_ready_event.set()
        else:
            with self.pins_frame_lock:
                self.pins_camera_frame = frame_pins
        
        if self.pins_flipped == "Yes":
            self.pins_camera_frame = cv2.flip(self.pins_camera_frame, -1)
            
    def RenderDifferenceImage(self, frame, reference_frame, source, mode):
        # Convert frame to grayscale
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

            # If the difference image is used for final ball tracking or debugging, mask the detection_bounds
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
        
        # Apply binary thresholding to segment the circle
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

    def ReceivePinsBuffer(self, buffer, fps_pins):

        self.pins_video_frame_buffer = buffer
        
        # Initialize the Pin Video Writer
        self.output_path_pins = os.path.join('recordings', f'pins_new_{self.lane_number}.mp4')
        self.out_pins = cv2.VideoWriter(self.output_path_pins, cv2.VideoWriter_fourcc(*'mp4v'), fps_pins, (self.pins_camera_width, self.pins_camera_height))

        # Process and save the frames in the buffer to the pins video file
        for frame in self.pins_video_frame_buffer:
            # Flip the frame if enabled
            if self.pins_flipped == "Yes":
                frame = cv2.flip(frame, -1)
            
            # Write each frame to the video file    
            self.out_pins.write(frame)

        # Release the Pin Video Writer
        self.out_pins.release()
        self.out_pins = None

        # Copy the exported pins video from recordings folder to videos to not overwrite it with the re-inialization of the file at the end of the shot
        self.output_path_pins_saved = os.path.join('videos', f'pins_new_{self.lane_number}.mp4')
        shutil.copy(self.output_path_pins, self.output_path_pins_saved)
    
    @pyqtSlot(object)
    def ReceiveTrackedFrame(self, frame):
        self.tracking_buffer.append(frame)

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

        centerlist = []
        self.frames_without_detection = 0
        self.detection_counter = 0
        self.detection_region = "start"
        cut_trigger = "inactive"
        self.lock_preshot_buffer = False
        self.started_pins_buffering = False
        
        # Define the variables holding the frames read from the cameras
        self.tracking_camera_frame = None
        self.pins_camera_frame = None
        
        # Define the variable holding all tracked frames
        self.tracking_buffer = []
        
        # Define the variable to hold the reference frame
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
        
        # Set events to signal once the cameras have been started and a frame has been read after intialization
        self.tracking_frame_ready_event = Event()
        self.pins_frame_ready_event = Event()
        
        # Start the camera workers
        self.tracking_camera_worker.start()
        self.pins_camera_worker.start()
        
        # Wait for both cameras to be ready and have their first frame captured
        self.tracking_frame_ready_event.wait()
        self.pins_frame_ready_event.wait()
        
        # Initialize the BallTracker with a frame
        with self.tracking_frame_lock:
            frame = self.tracking_camera_frame
        self.ball_tracker = TrackVideo(frame, self.lane_number, frame)
        self.ball_tracker.InitializeTracker()
        
        # Connect signals receiving tracked frames from the ball tracker
        signal_router.finalized_tracked_frame.connect(self.ReceivedTrackedFrame)
        
        # Emit a signal to show that the recorder is idle and ready to record
        self.recorder_status.emit("idle")

        # Define a variable to hold the time when the while loop starts and a list to store all times elapsed for the while loop
        start_time = None
        time_elapsed_list = []
        # Define a loop that runs, while the recorder is active
        while self.running:
            if start_time is not None:
                # Calculate the time elapsed for the previous run of the while loop
                time_elapsed = time.time() - start_time

                # Add this duration to the list
                time_elapsed_list.append(time_elapsed)

            # Set the start time to measure how long the while loop takes to run
            start_time = time.time()

            # Check if the preshot buffer is not locked (ball was not yet detected and the buffer should still be overwritten), then append the frame
            if not self.lock_preshot_buffer:
                with self.tracking_frame_lock:
                    frame = self.tracking_camera_frame
                self.preshot_video_frame_buffer.append(frame)
            
            # Show the debugging image if enabled and if a reference_frame has been set
            if self.show_debugging_image == "Yes" and self.reference_frame is not None:
                self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "debugging")
               
            # Check if we are past the ball entering the pins (cut_trigger active)
            if cut_trigger == "inactive":

                # Obtain the reference frame
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
                if self.frames_without_detection > 200:
                    self.detection_counter = 0

                # If enough centers at the start of the lane have been registered, switch the detection to the end, reset the counters and start recording
                elif self.detection_counter >= self.detection_threshold and self.detection_region == "start":
                    self.detection_region = "end"
                    self.detection_counter = 0
                    self.frames_without_detection = 0
                    centerlist=[]

                    # Lock the pre-shot buffer, so it is not overwritten anymore
                    self.lock_preshot_buffer = True

                    # Emit a signal to show that the recorder is now recording
                    self.recorder_status.emit("recording")

                    # Set the current pins camera frame to be the reference for score counting later
                    with self.pins_frame_lock:
                        self.pin_scorer_ref_frame = self.pins_camera_frame
                    
                # If not enough circles have been detected at the end of the lane, keep recording
                elif self.detection_counter < self.detection_threshold and self.detection_region == "end":
                    
                    # Generate the difference image to send to the tracker
                    with self.tracking_frame_lock:
                        frame = self.tracking_camera_frame
                    binary_image = self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "tracking")
                    
                    if not start_tracking_timer:
                        start_tracking_timer = True
                        start_time_ball_tracking = time.time()
                    
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

            # If the cut trigger is active and there are still frames to be recorded, record them as well as the ones for the pin camera
            elif cut_trigger == "active" and self.frames_after_shot > -1:
                if not self.started_pins_buffering:
                    self.start_pins_buffering.emit()
                    self.started_pins_buffering = True

                # Generate the difference image to send to the tracker
                with self.tracking_frame_lock:
                    frame = self.tracking_camera_frame
                binary_image = self.RenderDifferenceImage(frame, self.reference_frame, "ball tracking", "tracking")

                # Send the current frame to the ball tracker for tracking
                self.ball_tracker.TrackFrame(binary_image, frame)
                
                # Append the current pins camera image to the pins video buffer
                with self.pins_frame_lock:
                    pins_frame = self.pins_camera_frame

                # Show the debugging image if enabled and if a reference_frame has been set
                if self.show_debugging_image == "Yes" and self.pin_scorer_ref_frame is not None:
                    self.RenderDifferenceImage(pins_frame, self.pin_scorer_ref_frame, "pins", "debugging")

                if self.sweeper_detected == True:
                    # Increase the frame count after sweeper detection by one
                    self.frame_after_sweeper_detection += 1

                if self.sweeper_detected == False and self.pin_scorer_ref_frame is not None:
                    # Generate binary difference image of the pins frame
                    binary_image_pins = self.RenderDifferenceImage(pins_frame, self.pin_scorer_ref_frame,"pins", "tracking")

                    # Try to detect the sweeper in the binary image (a row of all white pixels)
                    self.DetectSweeper(binary_image_pins)

                # If time after sweeper detection has been reached, capture the read frame for the pins scoring
                if self.frame_after_sweeper_detection == self.time_pin_reading_after_sweeper:
                    self.pin_scorer_reading_frame = pins_frame

                # If the last frame is reached, save the last frame from the pin camera as image
                if self.frames_after_shot == 0:

                    # If no sweeper was detected, take the last frame as reading frame for the pin scoring
                    if self.sweeper_detected == False:
                        print("No sweeper was detected during this shot")
                        with self.pins_frame_lock:
                            self.pin_scorer_reading_frame = self.pins_camera_frame
                    
                    # Stop the timer for the ball tracking
                    if start_tracking_timer:
                        start_tracking_timer = False
                        stop_time_ball_tracking = time.time()
                        
                        # Calculate FPS of the Ball Tracking Camera
                        self.tracking_fps = int(len(self.tracking_buffer) / (stop_time_ball_tracking - start_time_ball_tracking))

                    # Emit a signal to show that the recorder is now saving the video files
                    self.recorder_status.emit("generating video")

                self.frames_after_shot -= 1

            # If the last frame to be recorded is reached, write the pin video and release both video files, then trigger the analysis and reset the variables for the next shot
            elif cut_trigger == "active" and self.frames_after_shot <= -1:
                if self.started_pins_buffering:
                    self.stop_pins_buffering.emit()
                    self.started_pins_buffering = False

                # Write pin score reading reference frame as pin image to be statically displayed
                cv2.imwrite('videos/pins_new_' + str(self.lane_number) + '.png', self.pin_scorer_ref_frame)
                
                # Initialize the Tracking Video Writer and write all the tracking frames to file
                self.output_path_tracking = os.path.join('videos', f'current_tracking_video_lane_{self.lane_number}.mp4')
                self.out = cv2.VideoWriter(self.output_path_tracking, cv2.VideoWriter_fourcc(*'mp4v'), self.tracking_fps, (self.tracking_camera_width, self.tracking_camera_height))
                
                for tracking_frame in self.tracking_buffer:
                    self.out.write(tracking_frame)
                                    
                # Emit a signal to show that the ball_tracker is now calculating all values of the shot and the pin score is read
                self.recorder_status.emit("calculating")
                
                # Perform all calculations and visualizations on the tracked frames
                self.ball_tracker.CalculateAndDraw()
                
                # Write the last frame a couple of times into the video
                for _ in range (self.tracking_fps * 0.3):
                    self.out.write(self.tracking_buffer[-1])
                
                # Copy the video file to the end location and release the writer
                shutil.copy(self.output_path_tracking, "videos/tracked_new_" + str(self.lane_number) + ".mp4")
                self.out.release()

                # Trigger the Score Reader
                self.scorer = PinScorer(self.pin_scorer_ref_frame, ast.literal_eval(self.config.get('Pin Scorer', 'pin_coordinates')))

                # Read the Score Reader
                standing_pins = self.scorer.PinsStillStanding(self.pin_scorer_reading_frame)

                signal_router.pins_standing_signal.emit(standing_pins)

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
                self.pins_video_frame_buffer.clear()
                self.pins_video_frame_buffer = deque(maxlen=self.frame_rate * self.export_video_buffer_length)
                self.frames_after_shot = self.frames_after_shot_restore
                self.lock_preshot_buffer = False
                self.frames_without_continuous_sweeper_detection = 0
                self.frames_after_sweeper_detection = 0
                self.sweeper_detected = False
                self.sweeper_count = 0
                self.pin_scorer_ref_frame = None
                self.pin_scorer_reading_frame = None
                frame = None
                pins_frame = None
                self.tracking_buffer = []

                # Re-initialize the Ball Tracker
                self.ball_tracker.InitializeTracker()

                # Emit a signal to show that the recorder is idle and ready to record
                self.recorder_status.emit("idle")

                # Calculate the average time it took for the while loop to run and reset the timer
                average_time_elapsed = sum(time_elapsed_list) / len(time_elapsed_list)
                print(f"Average while loop FPS: {int(1 / average_time_elapsed)}")

                start_time = None
                time_elapsed_list = []
        
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

        # Define the starting row in pixels
        row_index = 50

        # Loop through all pixel rows with a space of 10 pixels and check whether all pixels of that row are white
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
        if self.sweeper_count >= 3: # set the to the amount of continuous frames where the sweeper should be detected
            self.sweeper_detected = True
            print("Sweeper down detected")

    # Function to stop the recorder loop
    def StopMonitoring(self):
        self.running = False

# Worker class for frame capturing of both cameras
class FrameCaptureWorker(QThread):

    # Define signals to send the captured frame to the recorder worker
    tracking_frame_captured = pyqtSignal(object)
    pins_frame_captured = pyqtSignal(object)
    buffer_ready = pyqtSignal(list, int)

    def __init__(self, camera, stop_event, is_pins_camera=False):
        super().__init__()
        self.camera = camera
        self.stop_event = stop_event
        self.is_pins_camera = is_pins_camera
        self.running = True
        self.buffer = []
        self.buffering = False
        self.buffer_lock = Lock()

    def StartBuffering(self):
        # Start Buffering and reset the buffer
        self.buffering = True
        self.buffer = []
        self.start_time_buffer = time.time()

    def StopBuffering(self):
        with self.buffer_lock:
            # Stop buffering and emit the buffer to the RecorderWorker
            self.buffering = False
            print("Took ", int(time.time() - self.start_time_buffer), " sec. to buffer ", len(self.buffer), " frames yielding ", int(len(self.buffer) / (time.time() - self.start_time_buffer)), " FPS for the pins camera buffering")
            self.buffer_ready.emit(self.buffer, int(len(self.buffer) / (time.time() - self.start_time_buffer))) # Emit the buffer and the achieved FPS
            self.start_time_buffer = None

    def run(self):
        # Loop for capturing frames of the camera
        while self.running:

            ret, frame = self.camera.read()
            if not ret and not self.is_pins_camera:
                QMessageBox.critical(None, "Camera not readable", "Tracking Camera for this lane could not be accessed. Please ensure the camera is working and correctly selected in the settings.")
                break
            elif not ret and is_pins_camera:
                QMessageBox.critical(None, "Camera not readable", "Pins Camera for this lane could not be accessed. Please ensure the camera is working and correctly selected in the settings.")
                break

            # Emit the frame based on whether it's a tracking or pins camera frame
            if self.is_pins_camera:
                self.pins_frame_captured.emit(frame)
                
                # If the pin camera buffer is enabled, write the frame to the pin buffer
                with self.buffer_lock:
                    if self.buffering:
                        self.buffer.append(frame)

            else:
                self.tracking_frame_captured.emit(frame)

        # Set the event to signal successful stopping of the camera readings
        self.stop_event.set()

    def stop(self):
        self.running = False
