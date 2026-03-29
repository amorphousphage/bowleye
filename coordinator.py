# import used packages
import numpy as np
from collections import deque
import os
import cv2
import requests
import configparser
import ast
import time
import threading
from threading import Lock, Event, Thread
from ball_tracker import TrackVideo
from scorer import PinScorer
from camera_recorder import CameraRecorder
from http.server import BaseHTTPRequestHandler, HTTPServer
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

# Define the states of the coordinator depending on the activity
class State(Enum):
    IDLE = auto()           # No activity, watching for first ball
    TRACKING = auto()       # Actively tracking the ball
    PROCESSING = auto()     # Exporting Pin Video, Making calculations, Reading Score
    RESETTING = auto()      # Reset buffers, ready for next shot

class SwapBuffer:
    def __init__(self):
        self._frame = None
        self._lock = Lock()

    def set(self, frame):
        """Store the latest frame (overwrite previous)."""
        with self._lock:
            self._frame = frame

    def get(self):
        """Get the most recent frame (may be None if not yet set)."""
        with self._lock:
            return None if self._frame is None else self._frame.copy()

# Define a class for the coordinator of actions to run in a thread. The CoordinatorWorker acts as a timing and signal giver to various recording related threads
class CoordinatorWorker:
    def __init__(self, lane_number, status_callback=None):
        # Define a variable to hold the state of the coordinator
        self.state = State.IDLE

        # Obtain lane number for the Coordinator Worker
        self.lane_number = lane_number

        # Define a signal to pass the status of the recorder to the lane tab for the user to see
        self.status_callback = status_callback

        # Set the running mode to active
        self.running = False

        # Set Threads for ScoreReader and PinVideoExporter
        self.executor = ThreadPoolExecutor(max_workers=4)

    # --- Functions to control the coordinator ---

    # Function to set and update detection bounds region based on where the previous ball was detected
    def SetDetectionBounds(self, bounds, height, latest_ball_coordinate, modus):
        # Handle case if bounds are wrapped in outer array
        points = bounds[0] if bounds.ndim == 3 else bounds
        points = points.astype(np.int32)

        # Return the bottom part of the detection bounds with given height, if we are in start mode (not tracking yet, trying to detect first ball)
        if modus == "start":
            # Define the four corner points of the trapezoid
            p0, p1, p2, p3 = map(np.array, points)  # top-left, top-right, bottom-right, bottom-left

            # Compute vectors from bottom to top along left and right edges
            vec_left = p0 - p3
            vec_right = p1 - p2

            # Get unit directions
            dir_left = vec_left / np.linalg.norm(vec_left)
            dir_right = vec_right / np.linalg.norm(vec_right)

            # Move upward from bottom-left and bottom-right by height
            new_p3 = p3
            new_p2 = p2
            new_p0 = new_p3 + dir_left * height  # new top-left
            new_p1 = new_p2 + dir_right * height  # new top-right

            # Return the new smaller trapezoid
            return np.array([[new_p0, new_p1, new_p2, new_p3]], dtype=np.int32)

        # Else if no ball coordinate is given but we are tracking, return the whole detection_bounds, otherwise continue
        elif latest_ball_coordinate is None and modus == "tracking":
            return np.array([points], dtype=np.int32)

        # Define center of the ball
        center_x, bottom_y = latest_ball_coordinate
        width = int(2 * self.settings['max_diameter'])

        # Build the rectangle (bottom middle = ball coordinate)
        top_y = bottom_y - height
        rectangle = np.array([
            [center_x - width, top_y],    # top-left
            [center_x + width, top_y],    # top-right
            [center_x + width, bottom_y], # bottom-right
            [center_x - width, bottom_y], # bottom-left
        ], dtype=np.int32)

        # Clip rectangle to trapezoidal bounds using cv2
        trapezoid = points.reshape((-1, 1, 2))
        rect_poly = rectangle.reshape((-1, 1, 2))

        success, intersection = cv2.intersectConvexConvex(trapezoid.astype(np.float32), rect_poly.astype(np.float32))

        if success:
            return intersection.astype(np.int32).reshape(1, -1, 2)
        else:
            # No overlap, return the initial detection_bounds
            return np.array([points], dtype=np.int32)

    # Function to load all necessary settings
    def InitializeSettings(self):
        # Create the settings dictionary
        self.settings = {}

        # Load the configuration file
        config = configparser.ConfigParser()
        config_file = f'settings_lane_{self.lane_number}.cfg'
        if os.path.exists(config_file):
            config.read(config_file)
        else:
            print(f"No settings for lane {self.lane_number} were found. Please go to settings and choose the lane to autogenerate the settings file. Afterwards reboot the program.")
            return False
        # Populate the settings dictionary
        # Recorder settings
        self.settings['tracking_camera_path'] = config.get('Recorder', 'tracking_camera_path')
        self.settings['pins_camera_path'] = config.get('Recorder', 'pins_camera_path')
        self.settings['tracking_camera_resolution'] = (
            config.getint('Recorder', 'tracking_camera_x_resolution'),
            config.getint('Recorder', 'tracking_camera_y_resolution')
        )
        self.settings['pins_camera_resolution'] = (
            config.getint('Recorder', 'pins_camera_x_resolution'),
            config.getint('Recorder', 'pins_camera_y_resolution')
        )

        self.settings['fps_tracking_camera'] = config.getint('Recorder','fps_tracking_camera')
        self.settings['fps_pins_camera'] = config.getint('Recorder','fps_pins_camera')
        self.settings['pins_flipped'] = config.get('Recorder', 'pins_flipped')
        self.settings['reference_frame_distance'] = config.getint('Recorder', 'reference_frame_distance')
        self.settings['time_after_shot'] = round(config.getfloat('Recorder', 'time_after_shot'))
        self.settings['detection_threshold'] = config.getint('Recorder', 'detection_threshold')
        self.settings['detection_bounds'] = np.array(ast.literal_eval(config.get('Recorder', 'detection_bounds')), dtype=np.int32)
        self.settings['detection_bounds_height'] = config.getint('Recorder', 'detection_bounds_height')

        ## Calculate the start detection bounds for the recorder with a height in the same shape as the detection_bounds
        self.settings['recorder_start_bounds'] = self.SetDetectionBounds(self.settings['detection_bounds'], self.settings['detection_bounds_height'], None, "start")

        # Ball detection settings
        self.settings['blurred_kernel'] = config.getint('Ball Detection', 'blurred_kernel')
        self.settings['blurred_sigma'] = config.getint('Ball Detection', 'blurred_sigma')
        self.settings['binary_threshold'] = config.getint('Ball Detection', 'binary_threshold')
        self.settings['min_diameter'] = config.getint('Ball Detection', 'min_diameter')
        self.settings['max_diameter'] = config.getint('Ball Detection', 'max_diameter')
        self.settings['max_vertical_pixel_difference'] = config.getint('Ball Detection', 'max_vertical_pixel_difference')
        self.settings['max_horizontal_pixel_difference'] = config.getint('Ball Detection', 'max_horizontal_pixel_difference')
        self.settings['show_debugging_image'] = config.get('Ball Detection', 'show_debugging_image')
        self.settings['debugging_image_type'] = config.get('Ball Detection', 'debugging_image_type')
        self.settings['detection_bound_y_coordinate_near_foulline'] = self.settings['detection_bounds'][0][3][1]
        self.settings['top_detection_bounds_margin'] = config.getint('Ball Detection', 'top_detection_bounds_margin')
        self.settings['show_debugging_image'] = config.get('Ball Detection', 'show_debugging_image')

        # Ball Tracking Calculation settings
        self.settings['arrow_left'] = ast.literal_eval(config.get('Calculations', 'arrow_left'))
        self.settings['arrow_right'] = ast.literal_eval(config.get('Calculations', 'arrow_right'))
        self.settings['headpin_left'] = ast.literal_eval(config.get('Calculations', 'headpin_left'))
        self.settings['headpin_right'] = ast.literal_eval(config.get('Calculations', 'headpin_right'))

        self.settings['max_y_arrows_coordinate'] = config.getint('Calculations', 'max_y_arrows_coordinate')
        self.settings['min_y_arrows_coordinate'] = config.getint('Calculations', 'min_y_arrows_coordinate')
        self.settings['min_centers_for_calculation'] = config.getint('Calculations', 'min_centers_for_calculation')
        self.settings['amount_of_points'] = config.getint('Calculations', 'amount_of_points')
        self.settings['length_arrows_to_pins'] = config.getfloat('Calculations', 'length_arrows_to_pins')
        self.settings['visualize_minmax_arrow'] = config.get('Calculations', 'visualize_minmax_arrow')
        self.settings['minmax_arrow_distance'] = config.getint('Calculations', 'minmax_arrow_distance')
        self.settings['foulline_arrow_distance'] = config.getint('Calculations', 'foulline_arrow_distance')
        self.settings['foulline_excluded_points'] = config.getint('Calculations', 'foulline_excluded_points')

        # Pin scorer settings
        self.settings['time_pin_reading_after_sweeper'] = config.getfloat('Pin Scorer', 'time_pin_reading_after_sweeper')
        self.settings['pin_reading_size'] = config.getint('Pin Scorer', 'pin_reading_size')
        self.settings['pins_blurred_kernel'] = config.getint('Pin Scorer', 'pins_blurred_kernel')
        self.settings['pins_blurred_sigma'] = config.getint('Pin Scorer', 'pins_blurred_sigma')
        self.settings['pins_binary_threshold'] = config.getint('Pin Scorer', 'pins_binary_threshold')
        self.settings['pin_coordinates'] = config.get('Pin Scorer', 'pin_coordinates')
        return True

    # Function to initialize the capture from both cameras
    def InitializeCameras(self):
        # Set events to signal once the cameras have shut down when stopping the recorder to trigger release of the captures
        self.tracking_frame_stopped_event = Event()
        self.pins_frame_stopped_event = Event()

        # Set the locks for the tracking and pin camera buffering such that not more than one thread can access the shared buffer for the frames at once
        self.tracking_frame_lock = Lock()
        self.pins_frame_lock = Lock()

        # Create worker threads for both cameras to actually record and render difference images
        self.tracking_camera_worker = CameraRecorder(self.settings, self.tracking_camera_buffer, self.tracking_frame_stopped_event, is_pins_camera=False)
        self.pins_camera_worker = CameraRecorder(self.settings, self.pins_camera_buffer, self.pins_frame_stopped_event, is_pins_camera=True)

        return True

    # Set the variables for controlling the coordinator loop
    def SetCoordinatorVariables(self):
        ## Detection Variables
        # Define a variable to count frames without concurrent ball detections
        self.frames_without_detection = 0

        # Define a variable to count how many detections in a range were already recorded
        self.detection_counter = 0

        # Define the detection region where the coordinator expects the ball to be first when a shot is started and send this to the tracking camera worker
        self.detection_region = "start"
        self.detection_bounds = self.settings['recorder_start_bounds']

        # Define an empty list where to store ball positions and their size
        self.centerlist = []
        self.diameterlist = []


        ## Timing & Region Variables

        # Define a variable to know if the timer to measure ball speed has started
        self.start_tracking_timer = False

        # Define a variable to hold the time point when the cut trigger was activated
        self.cut_trigger_time = None

        # Define a variable to store whether the ball tracker has reported a ball close to the end of the detection bounds
        self.tracker_end_signalized = False

        ## Frame storring & Buffer Variables
        # Define a variable to lock the frames from being overwriten in the preshot buffer once a shot has started
        self.lock_preshot_buffer = False

        # Define the variable to hold the reference frame of the tracking camera
        self.reference_frame = None
        self.pins_reference_frame = None
        self.pins_binary_image = None

        # Define the variables holding the edited pins reference frame
        self.pins_reference_frame_edited = None

        # Define the variables to hold the pins reference and reading frame for score reading
        self.pin_scorer_ref_frame = None
        self.pin_scorer_reading_frame = None

        # Define the variable holding all tracked frames (from the ball tracker)
        self.tracked_frames_buffer = []

        ## Sweeper Detection Variables
        # Define a variable for a number of frames without sweeper detection to recognize false detections
        self.frames_without_continuous_sweeper_detection = 0

        # Define a variable to store sweeper detection
        self.sweeper_detected = False

        # Define a variable to count how many times a sweeper was detected
        self.sweeper_count = 0

        # Define a variable to store the time when the sweeper was detected
        self.time_sweeper_detected = None

        # Define the expected shape of the frame images for sanity checks to avoid glitchy/wrong frames
        self.expected_tracking_frame_shape = (
            self.settings['tracking_camera_resolution'][1],  # height (y)
            self.settings['tracking_camera_resolution'][0],  # width  (x)
            3  # BGR channels
        )


    # Function to initialize and Reset the Buffers
    def InitResetBuffers(self, mode):
        # Clear the buffers if they are being reset
        if mode == "reset":
            self.preshot_video_frame_buffer.clear()
            self.pins_video_frame_buffer.clear()

        # (Re-)Initialize the buffers
        self.preshot_video_frame_buffer = deque(maxlen=self.settings['reference_frame_distance'])

        self.tracking_camera_buffer = SwapBuffer()
        self.pins_camera_buffer = SwapBuffer()

        self.pins_video_frame_buffer = deque(maxlen=600)

        return True

    # Function to generate a difference image and a binary image depending on the need for it
    def RenderDifferenceImage(self, frame, reference_frame, reference_frame_edited, source, detection_bounds):
        # Mask a region of interest to the image if it comes from the ball tracking camera
        if source == "ball tracking":
            roi_frame = cv2.fillPoly(np.zeros_like(frame), detection_bounds, 255)
            # Apply the same region of interest to the reference
            roi_reference = cv2.fillPoly(np.zeros_like(reference_frame), detection_bounds, 255)

            # Gray the ROIs so they can be masked
            roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            roi_reference = cv2.cvtColor(roi_reference, cv2.COLOR_BGR2GRAY)

            # Apply the mask to the frame
            frame_cut_to_roi = cv2.bitwise_and(frame, frame, mask=roi_frame)

            # Apply the mask to the reference frame
            reference_frame_cut_to_roi = cv2.bitwise_and(reference_frame, reference_frame, mask=roi_reference)

            # Blur both frames
            frame_blurred = cv2.GaussianBlur(frame_cut_to_roi, (self.settings['blurred_kernel'], self.settings['blurred_kernel']), self.settings['blurred_sigma'])
            ref_blurred = cv2.GaussianBlur(reference_frame_cut_to_roi, (self.settings['blurred_kernel'], self.settings['blurred_kernel']), self.settings['blurred_sigma']) if reference_frame_edited is None else reference_frame_edited

            # Split into B, G, R channels
            b1, g1, r1 = cv2.split(frame_blurred)
            b2, g2, r2 = cv2.split(ref_blurred)

            # Compute absolute difference per channel
            diff_b = cv2.absdiff(b1, b2)
            diff_g = cv2.absdiff(g1, g2)
            diff_r = cv2.absdiff(r1, r2)

            # Combine differences of each channel
            diff_frame = cv2.max(cv2.max(diff_b, diff_g), diff_r)
            # Threshold the combined difference
            _, mask_to_use = cv2.threshold(diff_frame, self.settings['binary_threshold'], 255, cv2.THRESH_BINARY)

            # Broadcast the latest binary to the webserver if enabled
            if self.settings['show_debugging_image'] == 'Yes':
                if self.settings['debugging_image_type'] == 'Binary':
                    self.UploadImageToWebServer("binary",mask_to_use.copy())
                else:
                    self.UploadImageToWebServer("binary", diff_frame.copy())

            # return the finalized binary image and the edited reference for future re-use
            return mask_to_use

        # Do not mask a region of interest to the image if it comes from the pins camera
        elif source == "pins":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply a blur to the grayscale frame to reduce noise
            blurred = cv2.GaussianBlur(gray, (self.settings['blurred_kernel'], self.settings['blurred_kernel']), self.settings['blurred_sigma'])

            # If there is no saved edited reference image, create one
            if reference_frame_edited is None:
                reference_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
                # Add Blur to the Reference ROI image to reduce noise
                reference_frame_edited = cv2.GaussianBlur(reference_frame_gray, (self.settings['blurred_kernel'], self.settings['blurred_kernel']), self.settings['blurred_sigma'])

            # Compute the absolute difference between the current frame and the reference frame within the ROI
            diff_frame = cv2.absdiff(blurred, reference_frame_edited)

            # Apply binary thresholding to segment the differences (ball or pins or sweeper)
            _, binary = cv2.threshold(diff_frame, self.settings['binary_threshold'], 255, cv2.THRESH_BINARY)

            # Broadcast the latest pins binary to the webserver if enabled
            if self.settings['show_debugging_image'] == 'Yes':
                if self.settings['debugging_image_type'] == 'Binary':
                    self.UploadImageToWebServer("binary-pins", binary.copy())
                else:
                    self.UploadImageToWebServer("binary-pins", diff_frame.copy())

            # Return the finalized image and save the reference_frame_edited for future re-use
            return binary, reference_frame_edited

    # Function to detect circles in a edges detected difference image from the tracking camera
    def DetectBall(self, binary):
        # Set the variables to be used to find the ball
        max_diameter = 0
        best_y = None
        best_x_pair = None
        size_change_check = False

        # Set the shape of the image
        height, width = binary.shape

        # Get all y-values that contain any white pixels
        white_y_indices = np.where(np.any(binary == 255, axis=1))[0]

        # Only continue if white pixels were found
        if len(white_y_indices) > 0:

            # Define the cutoff to ignore top 40% of the white pixel vertical extent, which are the ones lower on the screen to avoid detecting shadows
            y_start = white_y_indices.min()
            y_end = int(white_y_indices.max() - 0.4 * (white_y_indices.max() - white_y_indices.min()))

            for y in range(y_start, y_end):
                white_xs = np.where(binary[y] == 255)[0]
                if len(white_xs) >= 2:
                    width_candidate = white_xs[-1] - white_xs[0]
                    if width_candidate > max_diameter:
                        max_diameter = width_candidate
                        best_y = y
                        best_x_pair = (white_xs[0], white_xs[-1])

            # If a widest line was found, determine the bottom center of the ball from it
            if best_y is not None and best_x_pair is not None:
                left_x, right_x = best_x_pair
                # Determine the middle of the distance of the white pixels (= The middle of the ball)
                middle_x = (left_x + right_x) // 2
                # Add half the width vertically to get to the bottom of the ball
                bottom_y = best_y + (max_diameter // 2)
                # Define the coordinates found
                ball_bottom_center = (middle_x, bottom_y)

                # Check if the detected circle fits the size of the ball and is not a horizontal glitch
                height_candidate = white_y_indices.max() - white_y_indices.min()
                aspect_ratio = max_diameter / max(height_candidate, 1)
                glitch_check = 0.5 < aspect_ratio < 3.0

                size_check = self.settings['max_diameter'] > max_diameter > self.settings['min_diameter']


                # If previous balls were detected, check if the diameter is not off by more than 20 %
                if self.diameterlist:
                    size_change_check = (
                        self.diameterlist[-1] * 0.8 < max_diameter< self.diameterlist[-1] * 1.2
                    )
                # If the ball_bottom is more towards the pins than the previous one and it passes the size & glitch check (and optionally the size_change_check), add it as a valid detection and draw the detected center into the frame
                if (size_check and glitch_check and (not self.centerlist or bottom_y < self.centerlist[-1][1]) and (not self.diameterlist or size_change_check)):
                    self.centerlist.append(ball_bottom_center)
                    self.diameterlist.append(max_diameter)
                    print("Ball detected")
                    print(f"size check: {size_check}")
                    print(f"glitch check: {glitch_check}, aspect_ratio: {round(aspect_ratio, 1)}, horizontal: {max_diameter}, vertical: {height_candidate}")
                else:
                    print(f"potential ball detected, but")
                    print("size_check: ", size_check)
                    print("glitch_check: ", glitch_check)
                    if self.diameterlist:
                        print("size_change_check: ", size_change_check)

    # Function to detect the sweeper in the pins camera
    def DetectSweeper(self, binary):

        # Set the sweeper count from last frame for comparison
        self.previous_sweeper_count = self.sweeper_count

        # Define the starting row in pixels (50 = 50 pixels away from the top of the image)
        row_index = 50

        # Loop through all pixel rows with a space of 10 pixels and check whether all pixels of that row are white (indicating a picture change done by the sweeper)
        while row_index < binary.shape[0] and self.sweeper_detected == False:
            if np.all(binary[row_index, :] == 255):
                self.sweeper_count += 1
                break
            # Move 10 pixels further down for the next detection
            row_index += 10

        # Include mechanism to clear wrong single sweeper detections after a while
        if self.sweeper_count == self.previous_sweeper_count:
            self.frames_without_continuous_sweeper_detection += 1

        elif self.sweeper_count > self.previous_sweeper_count:
            self.frames_without_continuous_sweeper_detection = 0

        if self.frames_without_continuous_sweeper_detection > 4:
            self.sweeper_count = 0

        # Check if enough concurrent sweeper counts have occured to define it as successfully detected
        if self.sweeper_count >= 3 and not self.sweeper_detected:
            self.sweeper_detected = True
            self.time_sweeper_detected = time.time()
            # For debugging, print if a sweeper was detected
            print("Sweeper down detected")
            ball_near_end_signal("Ball near the end detected")

    # --- Handler Functions to control the state of the coordinator ---
    def TransitionState(self, new_state):
        self.state = new_state

    # Define a function to execute while the coordinator is Idle (waiting for a new ball to enter the lane)
    def Handler_idle(self, frame):
        # Obtain the reference frame
        self.reference_frame = self.preshot_video_frame_buffer[0]

        # Render the binary difference image of the selected tracking camera frame
        binary_image = self.RenderDifferenceImage(frame, self.reference_frame, None, "ball tracking", self.detection_bounds)

        # Set the amount of circles detected from previous detection round
        self.centerlist_length = len(self.centerlist)

        # If no ball has been detected at the start yet (the region is still "start"), try to detect a ball in the starting region
        if self.detection_region == "start":
            # Detect circles in the current frame
            if binary_image is not None:
                self.DetectBall(binary_image)

            # If additional circles compared to last round were detected, increase the detection counter. If not, increase the counter for frames without detection
            if len(self.centerlist) > self.centerlist_length:
                self.detection_counter += 1
                self.frames_without_detection = 0

            else:
                self.frames_without_detection += 1

                # Reset the Detection Counter if enough frames without detection have passed (likely a false circle was detected in a previous timeframe)
                if self.frames_without_detection > 20:
                    self.detection_counter = 0
                    self.centerlist = []
                    self.centerlist_length = 0

        # If enough centers at the start of the lane have been registered, switch the detection to the end (handeled by the ball tracker), reset the counters and start recording
        if self.detection_counter >= self.settings['detection_threshold']:

            # Broadcast the latest binary that triggered recording to the webserver if enabled
            if self.settings['show_debugging_image'] == 'Yes':
                self.UploadImageToWebServer("trigger-binary", binary_image.copy())

            # Reset the detection counter and the frames without detection
            self.detection_counter = 0
            self.frames_without_detection = 0

            # Switch to detect the ball again at the end of the lane to signalize the end of a shot (done by the ball tracker)
            self.detection_region = "end"
            self.detection_bounds = self.SetDetectionBounds(self.settings['detection_bounds'], self.settings['detection_bounds_height'], self.centerlist[-1], "tracking")

            # Lock the pre-shot buffer, so it is not overwritten anymore, because the shot has started
            self.lock_preshot_buffer = True

            # Start the timer for the tracking camera
            if not self.start_tracking_timer:
                self.start_tracking_timer = True
                start_time_ball_tracking = time.time()

            # Start the pins camera if it is not running
            if not self.pins_camera_worker.recording_active_event.is_set():
                self.pins_camera_worker.recording_active_event.set()

            # Emit a signal to show that the recorder is now recording and switch the State of the coordinator to tracking
            self.status_callback("Shot recording started!")
            self.TransitionState(State.TRACKING)

    # Define a function to execute while the coordinator is Idle (waiting for a new ball to enter the lane)
    def Handler_tracking(self, frame):
        # Render the binary difference image of the selected tracking camera frame
        binary_image = self.RenderDifferenceImage(frame, self.reference_frame, None, "ball tracking", self.detection_bounds)

        # Send the current frame to the ball tracker for tracking and update the detection bounds in the tracking camera worker with what the ball tracker returns
        self.latest_ball_coordinate = self.ball_tracker.TrackFrame(binary_image, frame)
        self.detection_bounds = self.SetDetectionBounds(self.settings['detection_bounds'], self.settings['detection_bounds_height'], self.latest_ball_coordinate, "tracking")

        # If the ball tracker signalizes the ball at the end of the lane, start the cut_trigger and set the frames after sweeper detection
        if self.tracker_end_signalized and self.cut_trigger_time is None:
            self.cut_trigger_time = time.time()
            self.frame_after_sweeper_detection = 0

        # Check if the cut_trigger time has not fully elapsed yet
        if self.cut_trigger_time is not None and time.time() - self.cut_trigger_time < self.settings['time_after_shot']:

            # Ensure a frame is present in the pins camera buffer
            while True:

                if self.pins_camera_buffer.get() is not None:
                    break
                time.sleep(0.033) # wait 1/30 of a second to have a new frame added at 30 FPS

            # Obtain the frame in the pins camera buffer
            pins_frame = self.pins_camera_buffer.get()

            if self.pins_reference_frame is not None:
                self.pins_reference_frame = pins_frame
            # Add the pins_frame to the export buffer for later export
            self.pins_video_frame_buffer.append(pins_frame)

            # If no sweeper was detected yet, render the difference image for the pins to detect the sweeper
            if not self.sweeper_detected and self.pins_reference_frame is not None:
                self.pins_binary_image, self.pins_reference_frame_edited = self.RenderDifferenceImage(pins_frame, self.pins_reference_frame, self.pins_reference_frame_edited, "pins", None)

            # Try to detect the sweeper in the pins camera binary difference image and set the variables accordingly
            if self.pins_binary_image is not None and not self.sweeper_detected:
                self.DetectSweeper(self.pins_binary_image)

            # If not set, set the pins reference frame
            if self.pin_scorer_ref_frame is None:
                self.pin_scorer_ref_frame = pins_frame

            # If time after sweeper detection has been reached, capture the read frame for the pins scoring
            if self.time_sweeper_detected is not None and time.time() - self.time_sweeper_detected >= self.settings['time_pin_reading_after_sweeper']:
                self.pin_scorer_reading_frame = pins_frame
                # Reset the time so the frame is not captured multiple times
                self.time_sweeper_detected = None

        elif self.cut_trigger_time is not None and time.time() - self.cut_trigger_time >= self.settings['time_after_shot']:
            # If no sweeper was detected, take the last frame as reading frame for the pin scoring
            if self.sweeper_detected == False or self.pin_scorer_reading_frame is None:
                print("No sweeper was detected during this shot")

                # Obtain the reading frame as the current pin camera frame
                self.pin_scorer_reading_frame = self.pins_camera_buffer.get()

            # Emit a signal to show that the recorder is now saving the video files
            self.status_callback("Generating Pins Video and Reading the Score!")
            self.TransitionState(State.PROCESSING)

    def Handler_processing(self, frame):
        # Stop the camera workers
        self.tracking_camera_worker.stop()
        self.pins_camera_worker.stop()

        # Submit pin video export
        self.pin_video_path = os.path.join('videos', f'pins_new_{self.lane_number}.mp4')
        future_export = self.executor.submit(
            PinsVideoExporter().run,
            self.settings['fps_pins_camera'],
            self.pins_video_frame_buffer,
            self.pin_video_path
        )
        # Submit score reading
        future_score = self.executor.submit(
            ScoreReader().run,
            self.pin_scorer_ref_frame,
            self.pin_scorer_reading_frame,
            ast.literal_eval(self.settings['pin_coordinates']),
            self.settings
        )
        # Emit a signal to show that the ball_tracker is now calculating all values of the shot
        self.status_callback("Calculating the Trajectory and Shot Statistics!")

        # Perform all calculations and visualizations on the tracked frames
        self.latest_tracking = self.ball_tracker.CalculateTrackingData()
        self.UploadImageToWebServer("tracking-result", self.latest_tracking.copy())

        # Trigger the event loop to wait for the Pin Video Export Completion before starting to monitor for a new shot
        self.status_callback("Waiting for Pins Video Export to finish...")

        # Wait for PinsVideoExporter.run() to complete exporting
        future_export.result()

        # Wait for ScoreReader.run() to complete score reading
        future_score.result()

        # Upload the Pin Video to the Webapp
        self.UploadVideoToWebServer(self.pin_video_path)

        # Emit a signal to show that the recorder is now resetting itself for the next shot
        self.status_callback("Resetting the Ball Tracker!")
        self.TransitionState(State.RESETTING)

    def Handler_resetting(self, frame):
        # Reset all the necessary variables
        self.SetCoordinatorVariables()

        # Reset the buffers storring various camera frames
        self.InitResetBuffers("reset")

        # Re-initialize the Ball Tracker
        self.ball_tracker.InitializeTracker(self.settings)

        # Reinitialize the cameras
        if not self.InitializeCameras():
            print("Reinitializing Cameras unsuccessful!")
            return
        else:
            print("[CoordinatorWorker] Cameras reinitialized")

        # Restart the camera workers (as defined in the InitializeCameras)
        self.executor.submit(self.tracking_camera_worker.run)
        self.executor.submit(self.pins_camera_worker.run)

        # Start the recording for the tracking camera
        self.tracking_camera_worker.recording_active_event.set()

        # Emit a signal to show that the recorder is idle and ready to record
        self.status_callback("Ball Tracker is ready for the next Shot!")
        self.TransitionState(State.IDLE)

    # --- Functions for the Webapp
    def UploadImageToWebServer(self, name, img):
        ok, buf = cv2.imencode(".png", img)
        if not ok:
            return

        try:
            requests.post(
                "http://localhost:5000/debug/upload/" + name,
                files={"image": ("debug.png", buf.tobytes(), "image/png")},
                timeout=0.5
            )
        except Exception:
            pass

    def UploadVideoToWebServer(self, path_to_video):
        try:
            with open(path_to_video, "rb") as f:
                requests.post(
                    "http://localhost:5000/live/upload_video",
                    files={"video": ("shot.mp4", f, "video/mp4")},
                    timeout=2
                )
        except Exception as e:
            print("Video upload failed:", e)


    # Function to execute once the coordinator is activated
    def run(self):

        # Obtain the settings
        if not self.InitializeSettings():
            print("Initializing Settings unsuccessful!")
            return
        else:
            print("[CoordinatorWorker] Settings initialized")

        # Initialize the buffers storring various camera frames
        if not self.InitResetBuffers("init"):
            print("Buffers could not be initialized")
            return
        else:
            print("[CoordinatorWorker] Buffers initialized")

        # Initialize the cameras
        if not self.InitializeCameras():
            print("Initializing Cameras unsuccessful!")
            return
        else:
            print("[CoordinatorWorker] Cameras initialized")

        # Set the various Coordinator Variables
        self.SetCoordinatorVariables()
        # Start the camera workers (as defined in the InitializeCameras)
        self.executor.submit(self.tracking_camera_worker.run)
        self.executor.submit(self.pins_camera_worker.run)

        # Start the recording for the tracking camera
        self.tracking_camera_worker.recording_active_event.set()
        # Ensure a frame is present in the tracking camera buffer
        while True:
            if self.tracking_camera_buffer.get() is not None:
                break
            time.sleep(0.033) # wait 1/30 of a second to have a new frame added at 30 FPS

        # Initialize the BallTracker with a captured frame
        frame = self.tracking_camera_buffer.get()
        self.ball_tracker = TrackVideo(frame, self.lane_number, frame, ball_near_end_signal=lambda msg: setattr(self, 'tracker_end_signalized', True))
        self.ball_tracker.InitializeTracker(self.settings)

        # Emit a signal to show that the recorder is idle and ready to record
        self.status_callback("Ball Tracker is ready for the next Shot!")

        # Activate the running loop
        self.running = True

        # Define a loop that runs, while the recorder is active
        while self.running:

            # Obtain the current frame in the tracking camera buffer
            frame = self.tracking_camera_buffer.get()
            if frame is None:
                #print("frame is none")
                continue

            if frame.shape != self.expected_tracking_frame_shape:
                print("frame has an invalid shape")
                continue

            if np.std(frame) < 2:
                print("frame is not valid")
                continue

            # Check if the preshot buffer is not locked (ball was not yet detected and the preshot frame buffer should still be overwritten), then obtain and append the frame
            if not self.lock_preshot_buffer:
                self.preshot_video_frame_buffer.append(frame.copy())

            # Call the correct handler depending on state
            handler = getattr(self, f"Handler_{self.state.name.lower()}", None)

            if handler:
                handler(frame)
            else:
                print(f"No handler for state {self.state}")


    # Function to stop the recorder loop
    def stop(self):
         # Stop the camera workers
        self.tracking_camera_worker.stop()
        self.pins_camera_worker.stop()

        # Emit a signal to show that the recorder has been turned off
        self.status_callback("Ball Tracker and Recorder have been stopped!")

        # Exit the while loop
        self.running = False

# Create a class to read the pin score in a thread and pass the score to the lane_tab.py for display
class ScoreReader:
    def run(self, ref_frame, reading_frame, pin_coordinates, settings, parent=None):
        # Define the used variables
        self.settings = settings
        self.ref_frame = ref_frame
        self.reading_frame = reading_frame
        self.pin_coordinates = pin_coordinates

        # Initialize the PinScorer and process the frames
        scorer = PinScorer(self.pin_coordinates)
        fallen_pins = scorer.PinScoreReading(self.ref_frame, self.reading_frame, self.settings['pins_blurred_kernel'], self.settings['pins_blurred_sigma'], self.settings['pins_binary_threshold'], 255, self.settings['pin_reading_size'])

        # Send the Score to the webapp
        requests.post(
            "http://localhost:5000/live/upload_score",
            json={"fallen_pins": fallen_pins},
            timeout=2
        )

        # Print the result
        print(f"The Score is {len(fallen_pins)}. The Pins that fell are: ", fallen_pins)

        pass

# Generate a Thread to asynchronously export the Pins Video Buffer
class PinsVideoExporter:
    def run(self, fps_pins, pins_buffer, output_path):
        # Define the variables used
        self.fps_pins = fps_pins
        self.pins_buffer = pins_buffer
        self.output_path = output_path

        # Obtain the height and width of the frames to export
        pins_frame_height, pins_frame_width = self.pins_buffer[0].shape[:2]

        # Initialize a video writer
        out_pins = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps_pins, (pins_frame_width, pins_frame_height))

        # Export all frames
        for frame in self.pins_buffer:
            out_pins.write(frame)

        # Release the video writer
        out_pins.release()
        self.pins_buffer = None

        pass
