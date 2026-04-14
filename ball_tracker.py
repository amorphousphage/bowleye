#Import used packages
import cv2
import math
import json
import sys
import configparser
import ast
import os
import time
import shutil
import logging
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from collections import deque

class TrackVideo():
    
    def __init__(self, frame, lane_number, reference_frame, ball_near_end_signal=None):
        super().__init__()
        self.frame = frame
        self.lane_number = lane_number
        self.reference_frame = reference_frame
        self.ball_near_end_signal = ball_near_end_signal

        ################################
        # Working Function Definitions #
        ################################

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

                # Check if the detected circle fits the size of the ball
                size_check = self.settings['max_diameter'] > max_diameter > self.settings['min_diameter']


                # If previous balls were detected, check if the diameter is not off by more than 20 %
                if self.diameterlist:
                    size_change_check = (
                        self.diameterlist[-1] * 0.8 < max_diameter< self.diameterlist[-1] * 1.2
                    )
                # If the ball_bottom is more towards the pins than the previous one and it passes the size check, add it as a valid detection and draw the detected center into the frame
                if not self.centerlist or (self.centerlist and bottom_y < self.centerlist[-1][1]) and (not self.diameterlist or size_change_check):
                    self.centerlist.append(ball_bottom_center)
                    self.diameterlist.append(max_diameter)
                    cv2.circle(self.frame, ball_bottom_center, 2, (0, 255, 0), 2)
                    self.no_detection_frames = 0
                else:
                    self.no_detection_frames += 1

                # If the ball is towards the end of the lane, trigger the signal that the shot is ending
                if self.centerlist[-1][1] - self.settings['detection_bounds'][0][0][1] < self.settings['top_detection_bounds_margin'] and not self.end_triggered:
                    self.ball_near_end_signal("ball near the end detected")
                    logging.info("end detected")
                    self.end_triggered = True
            else:
                self.no_detection_frames += 1

            if self.no_detection_frames > 10 and not self.end_triggered:
                self.ball_near_end_signal("ball near the end detected")
                logging.info("no frames detected for 10 frames, stopping")
                self.end_triggered = True

        # Return the list of detected circles and the debug_frame
        return self.centerlist

    # Function to smooth the ball tracking fitted curve
    def SmoothCurve(self, points, window_size=3):

        # Create a window with all elements equal to 1/window_size
        window = np.ones(window_size) / window_size

        # Convolve the points with the window to apply moving average smoothing
        smoothed_points = np.convolve(points[:, 1], window, mode='valid')

        # Reconstruct the smoothed curve with x-coordinates from the original points
        smoothed_curve = np.column_stack((points[window_size // 2:-window_size // 2 + 1, 0], smoothed_points))

        return smoothed_curve.astype(int)

    def InitializeLaneHomography(self):
        # Pixel points from calibration (.cfg)
        arrow_left  = self.settings['arrow_left']   # [x, y]
        arrow_right = self.settings['arrow_right']
        pin_left    = self.settings['headpin_left']
        pin_right   = self.settings['headpin_right']

        pixel_pts = np.array([
            arrow_left,
            arrow_right,
            pin_left,
            pin_right
        ], dtype=np.float32)

        # Real-world lane coordinates: x in inches (0–42), y in meters from foul line
        real_pts = np.array([
            [0.0,  4.57],   # left at arrows
            [42.0, 4.57],   # right at arrows
            [0.0,  18.29],  # left at headpin
            [42.0, 18.29],  # right at headpin
        ], dtype=np.float32)

        self.H, _ = cv2.findHomography(pixel_pts, real_pts)

    def PixelToLaneCoordinates(self, x, y):
        pt = np.array([[[x, y]]], dtype=np.float32)  # shape (1,1,2)
        mapped = cv2.perspectiveTransform(pt, self.H)[0][0]
        lane_x_inches = float(mapped[0])   # 0–42
        lane_y_meters = float(mapped[1])   # ~4.57–18.29 in camera view
        return lane_x_inches, lane_y_meters

    def InchesToBoardsFromRight(self, lane_x_inches):
        # lane_x_inches: 0 = left, 42 = right
        # boards: 0 = right gutter, 39 = left gutter
        return (42.0 - lane_x_inches) / (42.0 / 39.0)

    def DrawBallPathImage(self, template_path, lane_positions):
        # Load template
        img = cv2.imread(template_path)
        if img is None:
            raise ValueError(f"Could not load template image: {template_path}")

        # Template geometry
        template_width = 150
        template_height = 570
        foul_line_y = 540
        headpin_y = 60

        lane_length_m = 18.29
        lane_width_in = 41.875

        # Calculate how many pixel per meter of lane length there are and how many pixel per inch of width there are
        px_per_meter = (foul_line_y - headpin_y) / lane_length_m
        px_per_inch = template_width / lane_width_in

        # Define a function to convert lane distance to pixels for drawing
        def LaneToPixel(lane_x_inches, lane_y_meters):
            x_px = int(lane_x_inches * px_per_inch)
            y_px = int(foul_line_y - lane_y_meters * px_per_meter)
            return x_px, y_px

        # Define the skid-hook-roll boundary phases
        skid_start = self.phase_indices["skid_start"]
        hook_start = self.phase_indices["hook_start"]
        hook_end = self.phase_indices["hook_end"]
        roll_end = self.phase_indices["roll_end"]

        # Define colors for the skid-hook-roll phasing
        COLOR_SKID = (0, 0, 255)        # red
        COLOR_HOOK = (0, 140, 255)      # dark orange
        COLOR_ROLL = (19, 69, 139)      # brown

        # Draw segments with phase colors
        for i in range(1, len(lane_positions)):
            x1, y1 = lane_positions[i - 1]
            x2, y2 = lane_positions[i]

            pt1 = LaneToPixel(x1, y1)
            pt2 = LaneToPixel(x2, y2)

            if i < hook_start:
                color = COLOR_SKID
            elif hook_start <= i <= hook_end:
                color = COLOR_HOOK
            else:
                color = COLOR_ROLL

            cv2.line(img, pt1, pt2, color, 3)

        return img



    ################################
    # Calling Function Definitions #
    ################################

    # Define a function that initializes all settings, functions and variables
    def InitializeTracker(self, settings):
        self.settings = settings

        #########################
        # Define used Variables #
        #########################

        # Initialize list of detected circle centers to store their coordinates and draw the path
        self.centerlist = []

        # Initialize a list to store the diameteres for size checking of detected circles
        self.diameterlist = []

        # Define global variables to store template images
        arrows_template = None

        #Gray the reference frame supplied to this function
        self.reference_frame_gray = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)

        # Define the region of interest (ROI) (used for ball and arrow detection)
        roi = cv2.fillPoly(np.zeros_like(self.reference_frame_gray), [self.settings['detection_bounds']], 255)

        # Cut the reference frame to the ROI
        self.reference_roi_gray = cv2.bitwise_and(self.reference_frame_gray, self.reference_frame_gray, mask=roi)

        # Add Blur to the Reference ROI image to reduce noise
        self.blurred_reference = cv2.GaussianBlur(self.reference_roi_gray, (self.settings['blurred_kernel'], self.settings['blurred_kernel']), self.settings['blurred_sigma'])

        # Calculate the mean vertical coordinate to calculate the reference line for the position of the arrows
        self.mean_y_arrows = (self.settings['max_y_arrows_coordinate'] + self.settings['min_y_arrows_coordinate']) // 2

        #Initialize the point_closest_to_gutter for the breakpoint calculation and storage
        self.point_closest_to_gutter = None

        # Define a variable to hold the calculated ball speed
        self.ball_speed = None

        # Define a variable that holds the time once a center past the arrow is detected (used for ball speed calculation)
        self.start_time = None

        # Define a variable that stores the length of the centerlist variable to compare between frames (to check whether new centers have been added or the ball has left the detection bounds). This is used to determine the end point for velocity calculation
        self.centerlist_length = 0

        # Define a variable to count consecutive frames without detected center (used for velocity calculation)
        self.frames_without_center = 0

        # Define a variable to hold the last frame
        self.last_frame = None

        # Define a variable to store whether the ball at the end of the lane signal has been triggered
        self.end_triggered = False
        self.no_detection_frames = 0

        # Initialize the Lane Homography
        self.InitializeLaneHomography()

    def TrackFrame(self, binary, frame):
        # Store the last frame for later recall showing the values (before drawing all centers on the frame, because they should not show in last_frame)
        self.frame = frame
        self.last_frame = self.frame.copy()
        self.binary = binary
        self.latest_ball_coordinate = None

        # Detect the ball
        self.DetectBall(self.binary)

        # Check if centers (circles) were detected
        if self.centerlist_length > 0:

            # Check if the latest center is close to the pins. If yes, calculate the ball speed
            if self.end_triggered and self.start_time and not self.ball_speed:

                # Calculate the ball speed, by calculating the seconds passed and dividing 13.5 m (distance from top arrow to pins) by it and multiplying by 3.6 to convert m/s to km/h
                self.ball_speed = round(self.settings['length_arrows_to_pins'] / (time.time() - self.start_time) * 3.6, 1)

            # Check if the latest center is after the minimal arrows, there is no ball speed calculated yet and the start time wasn't set
            if self.centerlist[-1][1] > self.settings['min_y_arrows_coordinate'] and not self.ball_speed and not self.start_time:

                # Define the start time of the speed measurement
                self.start_time = time.time()

        # Store current centerlist length for comparison with next frame
        self.centerlist_length = len(self.centerlist)

        # If it exists, set the latest ball coordinate
        if self.centerlist_length > 0:
            self.latest_ball_coordinate = self.centerlist[-1]

        # Return the latest ball coordinate to adapt detection bounds with it
        return self.latest_ball_coordinate

    def CalculateTrackingData(self):
        # Check if we have enough center points to initiate the calculations
        if not len(self.centerlist) > self.settings['min_centers_for_calculation']:
            # Write empty tracking data if we do not have enough centers for calculation
            self.tracking_data = {
                'ball_speed': "n/a",
                'position_at_foul_line': "n/a",
                'position_at_arrows': "n/a",
                'position_at_breakpoint': "n/a"
            }

            return None

        ################
        # Calculations #
        ################

        # Note: Some of these calculations are a not to easily understandable mathematical processes. See documentation for explanation (does not exist yet)

        ## ~~ Calculate a fitted line through all center points ~~ ##

        ###############################################
        # 1. Convert centerlist to lane coordinates   #
        ###############################################

        # Convert centerlist to numpy array for easier manipulation to fit a curve through all centers
        center_array = np.array(self.centerlist)

        # Convert the pixel coordinates to Lane Coordinates
        lane_points = []
        for (px, py) in center_array:
            lx, ly = self.PixelToLaneCoordinates(px, py)
            lane_points.append((lx, ly))

        lane_points = np.array(lane_points)

        # Sort by increasing y
        lane_points = lane_points[np.argsort(lane_points[:, 1])]

        ys = lane_points[:, 1]   # lane y in meters
        xs = lane_points[:, 0]   # lane x in inches

        ###############################################
        # 2. Fit a smoothing spline x(y)              #
        ###############################################

        # Smoothing factor s controls how smooth the curve is.
        smoothing = 2.0

        spline = UnivariateSpline(ys, xs, s=smoothing)

        ###############################################
        # 3. Evaluate the spline densely for drawing  #
        ###############################################

        y_dense = np.linspace(ys.min(), ys.max(), self.settings['amount_of_points'])
        x_dense = spline(y_dense)

        # Combine into lane_positions list
        lane_positions = list(zip(x_dense, y_dense))

        ###############################################
        # 4. Compute derivatives for phase detection  #
        ###############################################

        dx_dy  = spline.derivative(1)(y_dense)
        d2x_dy2 = spline.derivative(2)(y_dense)

        # Curvature proxy (absolute second derivative)
        curv = np.abs(d2x_dy2)

        # Optional smoothing to avoid jitter
        curv_smooth = gaussian_filter1d(curv, sigma=2)

        max_curv = np.max(curv_smooth)
        if max_curv < 1e-6:
            # Straight ball fallback
            hook_start_idx = 0
            hook_end_idx = len(curv_smooth) - 1
        else:
            skid_threshold = 0.20 * max_curv
            roll_threshold = 0.20 * max_curv

            # Hook start = first time curvature rises above threshold
            hook_start_idx = int(np.argmax(curv_smooth > skid_threshold))

            # Hook end = last time curvature is above threshold
            hook_end_idx = len(curv_smooth) - int(np.argmax(curv_smooth[::-1] > roll_threshold)) - 1

        # Store for drawing later
        self.phase_indices = {
            "skid_start": 0,
            "hook_start": hook_start_idx,
            "hook_end": hook_end_idx,
            "roll_end": len(y_dense) - 1
        }

        # --- Position at arrows (y ≈ 4.57 m) ---
        target_y_arrows = 4.57 #replace later with settings value
        idx_arrows = int(np.argmin([abs(ly - target_y_arrows) for (_, ly) in lane_positions]))
        lane_x_at_arrows = lane_positions[idx_arrows][0]
        ball_pos_at_arrows = round(self.InchesToBoardsFromRight(lane_x_at_arrows),1)

        # --- Position at foul line (y = 0 m) via regression in lane space ---
        ys = np.array([ly for (_, ly) in lane_positions])
        xs = np.array([lx for (lx, _) in lane_positions])

        # Use only points up to arrows for laydown estimation and launch angle calculation
        mask_up_to_arrows = ys <= target_y_arrows
        if np.count_nonzero(mask_up_to_arrows) >= 2:
            ys_fit = ys[mask_up_to_arrows]
            xs_fit = xs[mask_up_to_arrows]

            # Fit x = m*y + c
            m, c = np.polyfit(ys_fit, xs_fit, 1)
            x_at_foul = c  # y = 0 → x = c
            board_lay_down = round(self.InchesToBoardsFromRight(x_at_foul), 1)
            launch_slope = m
            launch_angle_deg = np.degrees(np.arctan(launch_slope))
        else:
            board_lay_down = "n/a"
            launch_angle_deg = "n/a"

        # --- Breakpoint: minimal distance to right gutter in boards ---
        boards_from_right_list = [self.InchesToBoardsFromRight(lx) for (lx, _) in lane_positions]
        min_idx = int(np.argmin(boards_from_right_list))
        min_distance_from_gutter = boards_from_right_list[min_idx]

        # Clamp small negatives to 0, larger negatives = gutter
        if 0 > min_distance_from_gutter >= -0.5:
            min_distance_from_gutter = 0.0
        elif min_distance_from_gutter < -0.5:
            min_distance_from_gutter = "Gutter"

        # Calculate Entry angle (near the pins)########
        entry_slope = dx_dy[-5]  # stable estimate
        entry_angle_deg = np.degrees(np.arctan(entry_slope))

        # --- Tracking data dict ---
        if self.ball_speed is None:
            self.ball_speed = "n/a"

        self.tracking_data = {
            'ball_speed': self.ball_speed,
            'position_at_foul_line': board_lay_down,
            'position_at_arrows': ball_pos_at_arrows,
            'position_at_breakpoint': round(min_distance_from_gutter, 1) if isinstance(min_distance_from_gutter, (int,float)) else min_distance_from_gutter,
            'lane_positions': lane_positions,
            'launch_angle': round(launch_angle_deg, 1) if isinstance(launch_angle_deg, (int, float)) else launch_angle_deg,
            'entry_angle': round(entry_angle_deg, 1) if isinstance(entry_angle_deg, (int, float)) else entry_angle_deg,
            }

        logging.info(
            "\n--- Tracking Data ---\n"
            f"Ball Speed: {self.tracking_data['ball_speed']} km/h\n"
            f"Position @ Foul Line: {self.tracking_data['position_at_foul_line']}\n"
            f"Position @ Arrows: {self.tracking_data['position_at_arrows']}\n"
            f"Position @ Breakpoint: {self.tracking_data['position_at_breakpoint']}\n"
            f"Launch Angle: {self.tracking_data['launch_angle']}\n"
            f"Entry Angle: {self.tracking_data['entry_angle']}\n"
            "---------------------"
        )


        # Draw the calculations on the frame
        self.tracking_image = self.DrawBallPathImage(template_path="templates/lane.png", lane_positions=lane_positions)

        return self.tracking_image
