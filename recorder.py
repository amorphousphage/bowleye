# import used packages
import numpy as np
from collections import deque
import os
import cv2
import configparser
import ast
import shutil
from ball_tracker import TrackVideo
from scorer import PinScorer
from signal_router import signal_router
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QMessageBox

# Generate a class for the recorder to run in a thread
class RecorderWorker(QThread):
    # Define a signal to pass the status of the recorder to the lane tab for the user to see
    recorder_status = pyqtSignal(str)

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
        self.frame_rate = config.getint('Recorder', 'recorder_frame_rate')
        self.frame_rate_pins = config.getint('Recorder', 'pins_frame_rate')
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
        self.export_video_frame_buffer = deque(maxlen=self.frame_rate * self.export_video_buffer_length)

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

        self.show_debugging_image = self.config.get('Ball Detection', 'show_debugging_image')
        self.debugging_image_type = self.config.get('Ball Detection', 'debugging_image_type')

        return True

    # Function to initialize the Video Writer objects for both cameras
    def InitializeVideoWriters(self):

        # Set the path for where the recorder should store the tracking camera and pins raw file and define the Video Writers
        self.output_path = os.path.join('recordings', f'recorded_new_lane_{self.lane_number}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(self.output_path, fourcc, self.frame_rate, (self.tracking_camera_width, self.tracking_camera_height))

        self.output_path_pins = os.path.join('recordings', f'pins_new_{self.lane_number}.mp4')
        self.out_pins = cv2.VideoWriter(self.output_path_pins, fourcc, self.frame_rate_pins, (self.pins_camera_width, self.pins_camera_height))

        return True

    # Function to release the Video Writers
    def ReleaseVideoWriters(self):
        # Release the Tracking Camera Video Writer
        if self.out:
            self.out.release()
            self.out = None

        # Release the Pins Camera Video Writer
        if self.out_pins:
            self.out_pins.release()
            self.out_pins = None

    # Function to initialize the capture from both cameras
    def InitializeCameras(self):
        # Initialize Tracking Camera with defined properties
        self.cap = cv2.VideoCapture(self.config.get('Recorder', 'tracking_camera_path'))
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FPS, self.frame_rate)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.tracking_camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.tracking_camera_height)

        # Initialize Pins Camera with defined properties
        self.cap_pins = cv2.VideoCapture(self.config.get('Recorder', 'pins_camera_path'))
        self.cap_pins.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap_pins.set(cv2.CAP_PROP_FPS, self.frame_rate_pins)
        self.cap_pins.set(cv2.CAP_PROP_FRAME_WIDTH, self.pins_camera_width)
        self.cap_pins.set(cv2.CAP_PROP_FRAME_HEIGHT, self.pins_camera_height)

        return True

    # Function to execute once the recorder is called
    def run(self):
        # Obtain the settings
        if not self.InitializeSettings():
            self.finished.emit()
            return

        # Initialize the cameras
        if not self.InitializeCameras():
            self.finished.emit()
            return

        # Initialize the Video Writers
        if not self.InitializeVideoWriters():
            self.finished.emit()
            return

        # Set control variables for triggering and performing the recording/analysis
        centerlist = []
        self.frames_without_detection = 0
        self.detection_counter = 0
        self.detection_region = "start"
        cut_trigger = "inactive"
        self.lock_preshot_buffer = False
        
        # Initialize the BallTracker with a frame
        ret, frame = self.cap.read()
        self.ball_tracker = TrackVideo(frame, self.lane_number, frame)
        self.ball_tracker.InitializeTracker()

        # Emit a signal to show that the recorder is idle and ready to record
        self.recorder_status.emit("idle")

        # Define a loop that runs, while the recorder is active
        while self.running:

            # Try to read the current camera image
            ret, frame = self.cap.read()
            if not ret:
                QMessageBox.critical(None, "Camera not readable", "Tracking Camera for lane " + str(
                self.lane_number) + " could not be accessed. Please ensure the camera is working and correctly selected in the settings.")
                break
            # Check if the preshot buffer is not locked (ball was not yet detected and the buffer should still be overwritten), then append the frame
            if not self.lock_preshot_buffer:
                self.preshot_video_frame_buffer.append(frame)

            # Check if we are past the ball entering the pins (cut_trigger active)
            if cut_trigger == "inactive":

                # Obtain the reference frame
                if len(self.preshot_video_frame_buffer) < self.reference_frame_distance:
                    self.reference_frame = self.preshot_video_frame_buffer[0]
                else:
                    self.reference_frame = self.preshot_video_frame_buffer[-self.reference_frame_distance]

                # Gray the reference frame and apply the corresponding detection region
                reference_frame_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)
                if self.detection_region == "start":
                    roi_reference = cv2.fillPoly(np.zeros_like(reference_frame_gray), self.start_detection_bounds, 255)
                elif self.detection_region == "end":
                    roi_reference = cv2.fillPoly(np.zeros_like(reference_frame_gray), self.end_detection_bounds, 255)

                reference_roi_gray = cv2.bitwise_and(reference_frame_gray, reference_frame_gray, mask=roi_reference)

                # Blur the reference image
                blurred_reference = cv2.GaussianBlur(reference_roi_gray, (self.ksize, self.ksize), self.sigma)

                # Set the amount of circles detected from previous detection round
                self.centerlist_length = len(centerlist)

                # Detect circles in the current frame
                centerlist = self.DetectCircles(frame, blurred_reference, centerlist)

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

                # If not enough circles have been detected at the end of the lane, keep recording
                elif self.detection_counter < self.detection_threshold and self.detection_region == "end":

                    # Send the current frame to the ball tracker for tracking
                    self.ball_tracker.TrackFrame(frame)

                # If enough circles at the end of the lane have been detected, activate the cut trigger (run out period to record the pins camera) and keep recording
                elif self.detection_counter >= self.detection_threshold and self.detection_region == "end":
                    cut_trigger = "active"
                    self.frames_after_shot -= 1
                    self.current_pin_frame = 0

                    # Send the current frame to the ball tracker for tracking
                    self.ball_tracker.TrackFrame(frame)

            # If the cut trigger is active and there are still frames to be recorded, record them as well as the ones for the pin camera
            elif cut_trigger == "active" and self.frames_after_shot > -1:

                # Send the current frame to the ball tracker for tracking
                self.ball_tracker.TrackFrame(frame)

                # Obtain the frame from the pins camera
                ret_pins, frame_pins = self.cap_pins.read()
                if not ret_pins:
                    QMessageBox.critical(None, "Camera not readable", "Pins Camera for lane " + str(
                self.lane_number) + " could not be accessed. Please ensure the camera is working and correctly selected in the settings.")
                    break

                # Flip the frame if defined so in settings (camera is mounted upside down)
                if self.pins_flipped == "Yes":
                    frame_pins_flipped = cv2.flip(frame_pins, -1)
                    self.out_pins.write(frame_pins_flipped)
                    
                    # Obtain the reference and reading frame for the pin scorer if conditions apply:
                    if self.current_pin_frame == 0:
                        self.pin_scorer_ref_frame = frame_pins_flipped
                        
                    elif self.current_pin_frame == self.config.getint('Pin Scorer', 'time_pin_reading_after_start'):
                        self.pin_scorer_reading_frame = frame_pins_flipped
                        
                else:
                    self.out_pins.write(frame_pins)

                    # Obtain the reference and reading frame for the pin scorer if conditions apply:
                    if self.current_pin_frame == 0:
                        self.pin_scorer_ref_frame = frame_pins
                    elif self.current_pin_frame == self.config.getint('Pin Scorer', 'time_pin_reading_after_start'):
                        self.pin_scorer_reading_frame = frame_pins

                # If the last frame is reached, save the last frame from the pin camera as image
                if self.frames_after_shot == 0:
                    # Emit a signal to show that the recorder is now saving the video files
                    self.recorder_status.emit("generating video")
                    
                    # Write the last frame of the pins video as pin image to be statically displayed
                    cv2.imwrite('videos/pins_new_' + str(self.lane_number) + '.png', frame_pins_flipped)


                self.current_pin_frame += 1
                self.frames_after_shot -= 1

            # If the last frame to be recorded is reached release both video files, then trigger the analysis and reset the variables for the next shot
            elif cut_trigger == "active" and self.frames_after_shot <= -1:
                # Release the Video Writers
                self.ReleaseVideoWriters()
                
                # Copy the exported pins video from recordings folder to videos to not overwrite it with the re-inialization of the file at the end of the shot
                self.output_path_pins_saved = os.path.join('videos', f'pins_new_{self.lane_number}.mp4')
                shutil.copy(self.output_path_pins, self.output_path_pins_saved)
                
                # Emit a signal to show that the recorder is now tracking the shot
                self.recorder_status.emit("tracking")
                
                # Perform all calculations and visualizations on the tracked frames
                self.ball_tracker.CalculateAndDraw()

                # Trigger the Score Reader
                self.scorer = PinScorer(self.pin_scorer_ref_frame, ast.literal_eval(self.config.get('Pin Scorer', 'pin_coordinates')))

                # Test Pin Detection without Canny
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
                self.export_video_frame_buffer.clear()
                self.export_video_frame_buffer = deque(maxlen=self.frame_rate * self.export_video_buffer_length)
                self.pins_video_frame_buffer.clear()
                self.pins_video_frame_buffer = deque(maxlen=self.frame_rate * self.export_video_buffer_length)
                self.frames_after_shot = self.frames_after_shot_restore
                self.lock_preshot_buffer = False

                # Re-initialize the Ball Tracker
                self.ball_tracker.InitializeTracker()

                #  Re-initialize the VideoWriters to have them ready for recording the next shot
                self.InitializeVideoWriters()

                # Emit a signal to show that the recorder is idle and ready to record
                self.recorder_status.emit("idle")

        # Release the cameras after the recorder is stopped
        self.cap.release()
        self.cap_pins.release()

        # Emit a signal to show that the recorder has been turned off
        self.recorder_status.emit("recorder offline")

    # Function to detected circles for the recorder
    def DetectCircles(self, frame, reference_frame, centerlist):

        # Gray the current frame and apply the detection area to it
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.detection_region == "start":
            roi = cv2.fillPoly(np.zeros_like(gray), self.start_detection_bounds, 255)
        elif self.detection_region == "end":
            roi = cv2.fillPoly(np.zeros_like(gray), self.end_detection_bounds, 255)

        roi_gray = cv2.bitwise_and(gray, gray, mask=roi)

        # Blur the current frame
        blurred = cv2.GaussianBlur(roi_gray, (self.ksize, self.ksize), self.sigma)

        # Calculate the difference to the reference frame and apply binary thresholding to it
        diff_frame = cv2.absdiff(blurred, reference_frame)
        _, binary = cv2.threshold(diff_frame, self.binary_threshold, self.binary_maximum, cv2.THRESH_BINARY)

        if self.show_debugging_image == "Yes":
            if self.debugging_image_type == "Binary":
                # Show the binary image in a new window
                cv2.imshow("Debugging - Binary Image", binary)
            elif self.debugging_image_type == "Difference Only":
                # Show the difference image in a new window
                cv2.imshow("Debugging - Difference Only Image", diff_frame)

            cv2.waitKey(50)  # Use a small delay to allow image to render

        # Try to find circles in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # For each identified circle, if it fulfills the criteria of belonging to the ball, add it to the centerlist
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center_bottom = (int(x), int(y + radius))

            if self.min_radius < radius < self.max_radius:
                centerlist.append(center_bottom)
        return centerlist

    # Function to stop the recorder loop
    def StopMonitoring(self):
        self.running = False
