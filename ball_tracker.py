#Import used packages
import cv2
import math
import json
import sys
import configparser
import ast
import os
import time
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from collections import deque
from PyQt5.QtWidgets import QMessageBox
from signal_router import signal_router

def TrackVideo(video_path, lane_number, reference_frame):
    # Load Settings
    config = configparser.ConfigParser()
    if os.path.exists(f'settings_lane_{lane_number}.cfg'):
        config.read(f'settings_lane_{lane_number}.cfg')
    else:
        QMessageBox.critical(None, "No Settings found",
                             "No settings for lane " + str(lane_number) + " were found. Please go to settings and choose the lane to autogenerate the settings file. Afterwards reboot the programm.")
        return

    ########################
    # Function Definitions #
    ########################

    # Function to detect arrows in a frame
    def DetectArrows(frame):
        global arrows_template

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Define paths to arrow template image
        arrows_template_path = f'templates/arrows_template_lane_{lane_number}.png'

        # Load arrow template
        arrows_template = cv2.imread(arrows_template_path, cv2.IMREAD_GRAYSCALE)

        # Check if template was loaded successfully
        if arrows_template is None:
            print("Error: Could not load the arrows template image")
            exit()

        # Extract region of interest (ROI) based on the defined detection bounds
        roi = cv2.fillPoly(np.zeros_like(gray), detection_bounds, 255)

        # Mask the grayscale frame with the ROI
        gray_roi = cv2.bitwise_and(gray, gray, mask=roi)

        # Match templates using template matching within the ROI
        arrow_match = cv2.matchTemplate(gray_roi, arrows_template, cv2.TM_CCOEFF_NORMED)

        # Define threshold for matches
        arrow_threshold = config.getfloat('Lane Setup', 'arrow_threshold')

        # Find locations of matches above threshold
        arrow_locations = np.where(arrow_match >= arrow_threshold)

        # Store the positions of arrows detected in the first frame
        arrow_positions = []
        for pt in zip(*arrow_locations[::-1]):
            arrow_positions.append(pt)

        return arrow_positions, arrows_template

    #Function to detect the ball
    def DetectBall(frame,reference_frame):
        # Define a global variable to hold the debugging image
        global debug_frame

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Mask the grayscale frame with the ROI
        roi_gray = cv2.bitwise_and(gray, gray, mask=roi)

        # Apply a blur to the grayscale frame to reduce noise
        blurred = cv2.GaussianBlur(roi_gray,(config.getint('Ball Detection', 'blurred_kernel'), config.getint('Ball Detection', 'blurred_kernel')), config.getint('Ball Detection', 'blurred_sigma'))

        # Compute the absolute difference between the current frame and the reference frame within the ROI
        diff_frame = cv2.absdiff(blurred, reference_frame)

        # Apply binary thresholding to segment the circle
        _, binary = cv2.threshold(diff_frame, config.getint('Ball Detection','binary_threshold'), config.getint('Ball Detection','binary_max'), cv2.THRESH_BINARY)

        # Find contours in the binary mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Define the minimum and maximum radius of the circle to be detected
        min_radius = config.getint('Ball Detection','min_radius')
        max_radius = config.getint('Ball Detection','max_radius')

        # Calculate maximum y-axis starting coordinate from the detection_bounds (y-axis coordinate of the roi closes to the foul line)
        starting_y_coordinate = max(point[1] for detection_bound in detection_bounds for point in detection_bound)

        # Set thresholds for max distance between two centers vertically (y-axis) and horizontally (x-axis)
        y_coordinate_threshold = config.getint('Ball Detection','max_vertical_pixel_difference') #Threshold for max y-axis distance between two points or first point to roi border (towards the foul line)
        x_coordinate_threshold = config.getint('Ball Detection','max_horizontal_pixel_difference') #Threshold for max x-axis distance between two points

        # Create a list to store detected circles
        detected_circles = []

        # Loop over the contours
        for contour in contours:
            # Find the center and radius of the enclosing circle for the contour
            (x, y), radius = cv2.minEnclosingCircle(contour)

            #Define the center of the circle
            center = (int(x), int(y))

            # Define the bottom of the circle (the area where the ball or circle touches the lane)
            center_bottom = (int(x), int(y + radius))

            # Check if circle in question belongs to the ball motion (with the restrictions/thresholds defined above) and if yes, draw it
            if min_radius < radius < max_radius and (not centerlist and starting_y_coordinate + y_coordinate_threshold >= center_bottom[1] >= starting_y_coordinate - y_coordinate_threshold or (centerlist and centerlist[-1][1] - y_coordinate_threshold < center_bottom[1] < centerlist[-1][1] and centerlist[-1][0] - x_coordinate_threshold < center_bottom[0] < centerlist[-1][0] + x_coordinate_threshold)):

                # Draw the circle and its center on the original image
                cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                cv2.circle(frame, center_bottom, 2, (0, 0, 255), 3)

                #Add the validated center coordinates to a list containing all identified valid center points across frames
                centerlist.append(center_bottom)

        # show any image used in calculation of circles rather than showing the true video image (for debugging purposes)
        if config.get('Ball Detection', 'show_debugging_image') == "Yes":

            # show the binary image if set in the settings
            if config.get('Ball Detection', 'debugging_image_type') == "Binary":
                debug_frame = binary[:, min_x_videoexport:max_x_videoexport]

            # show the difference image if set in settings
            elif config.get('Ball Detection', 'debugging_image_type') == "Difference only":
                debug_frame = diff_frame[:, min_x_videoexport:max_x_videoexport]

            # Else show the normal image
            else:
                debug_frame = frame[:, min_x_videoexport:max_x_videoexport]
        # Else define the normal image as debug image
        else:
            debug_frame = frame[:, min_x_videoexport:max_x_videoexport]

        # Return the list of detected circles and the debug_frame
        return detected_circles, centerlist, debug_frame


    #Function to interpolate a point anywhere on the trapeze defining the detection_bounds or lane_bounds
    def InterpolatePoint(p1, p2, y):
        x1, y1 = p1
        x2, y2 = p2
        x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
        return int(x), int(y)

    #Function to smooth the ball tracking fitted curve
    def SmoothCurve(points, window_size=3):

        # Create a window with all elements equal to 1/window_size
        window = np.ones(window_size) / window_size

        # Convolve the points with the window to apply moving average smoothing
        smoothed_points = np.convolve(points[:, 1], window, mode='valid')

        # Reconstruct the smoothed curve with x-coordinates from the original points
        smoothed_curve = np.column_stack((points[window_size//2:-window_size//2+1, 0], smoothed_points))

        return smoothed_curve.astype(int)

    #########################
    # Define used Variables #
    #########################

    # Initialize a variable to store the debug_frame if needed
    if config.get('Ball Detection', 'show_debugging_image') == "Yes":
        debug_frame = None

    # Initialize list of detected circle centers to store their coordinates and draw the path
    centerlist = []

    # Define global variables to store template images
    arrows_template = None

    # Obtain the detection and lane bounds from the settings as indicated by the user
    detection_bounds_str = config.get('Lane Setup', 'detection_bounds')
    detection_bounds = np.array(ast.literal_eval(detection_bounds_str), dtype=np.int32)
    lane_bounds_str = config.get('Lane Setup', 'lane_bounds')
    lane_bounds = np.array(ast.literal_eval(lane_bounds_str), dtype=np.int32)

    #Gray the reference frame supplied to this function
    reference_frame_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)

    # Define the region of interest (ROI) (used for ball and arrow detection)
    roi = cv2.fillPoly(np.zeros_like(reference_frame_gray), [detection_bounds], 255)

    # Cut the reference frame to the ROI
    reference_roi_gray = cv2.bitwise_and(reference_frame_gray, reference_frame_gray, mask=roi)

    # Add Blur to the Reference ROI image to reduce noise
    blurred_reference = cv2.GaussianBlur(reference_roi_gray, (config.getint('Ball Detection', 'blurred_kernel'), config.getint('Ball Detection', 'blurred_kernel')), config.getint('Ball Detection', 'blurred_sigma'))

    # Detect arrows in the first frame
    arrow_positions, arrows_template = DetectArrows(reference_frame)

    # Calculate the max and min vertical coordinate for the arrows detected
    max_y_arrows = max(arrow[1] for arrow in arrow_positions)
    min_y_arrows = min(arrow[1] for arrow in arrow_positions)

    # Calculate the mean vertical coordinate to calculate the reference line for the position of the arrows
    mean_y_arrows = (max_y_arrows + min_y_arrows) // 2

    # Define the x-coordinates where the mean_y_arrows line intersects the lane bounds (these coordinates will be used for calculation and line drawing)
    left_bound_at_arrows = InterpolatePoint(lane_bounds[0][0], lane_bounds[0][3], mean_y_arrows)
    right_bound_at_arrows = InterpolatePoint(lane_bounds[0][1], lane_bounds[0][2], mean_y_arrows)

    #Initialize the point_closest_to_gutter for the breakpoint calculation and storage
    point_closest_to_gutter = None

    # Load video file
    cap = cv2.VideoCapture(video_path)

    # Check if video file opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()

    # Define the bounds for the video export format
    min_x_videoexport = detection_bounds[0][3][0] - config.getint('Video Export', 'margins_video_export')
    max_x_videoexport = detection_bounds[0][2][0] + config.getint('Video Export', 'margins_video_export')

    # Get video properties (for video export)
    frame_width = int(max_x_videoexport - min_x_videoexport)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = "videos/tracked_new_" + str(lane_number) + ".mp4"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    # Define the VideoWriter for the debugging video if necessary
    if config.get('Ball Detection', 'show_debugging_image') == "Yes":
        output_debug_video_path = "videos/debug_new_" + str(lane_number) + ".mp4"
        debug_out = cv2.VideoWriter(output_debug_video_path, fourcc, fps, (frame_width, frame_height))

    # Create a variable to check whether first center was detected and the buffer was already written to the video (in order to not do it multiple times)
    first_center_detected = False

    # Define a variable to see if the closing frames have been triggered (used for initiating the stop of the frame processing)
    cut_trigger = "inactive"

    # Define a variable to hold the calculated ball speed
    ball_speed = None

    # Define a variable that holds the time once a center past the arrow is detected (used for ball speed calculation)
    start_time = None

    # Define a variable that stores the length of the centerlist variable to compare between frames (to check whether new centers have been added or the ball has left the detection bounds). This is used to determine the end point for velocity calculation
    centerlist_length = 0

    # Define a variable to count consecutive frames without detected center (used for velocity calculation)
    frames_without_center = 0


    ####################
    # Video Processing #
    ####################

    #Loop through all frames of the video
    while cap.isOpened():

        # Try to read the current frame
        ret, frame = cap.read()

        # Check if frame was read successfully
        if not ret:
            break

        # Detect the ball
        DetectBall(frame, blurred_reference)

        # Write the debugging frame if enabled
        if config.get('Ball Detection', 'show_debugging_image') == "Yes":
            # Convert the grayscale debug_frame to a three channel color frame and write it to the video
            debug_out.write(cv2.cvtColor(debug_frame, cv2.COLOR_GRAY2BGR))

        # Visualize arrows in the frame
        for arrow_pos in arrow_positions:
            cv2.rectangle(frame, arrow_pos, (arrow_pos[0] + arrows_template.shape[1], arrow_pos[1] + arrows_template.shape[0]), (150, 150, 150), 2)

        # Draw a horizontal line at the mean vertical value for the arrows in the region of interest (roi)
        cv2.line(frame, left_bound_at_arrows, right_bound_at_arrows, (255, 0, 0), 4)

        # Store the last frame for later recall showing the values (before drawing all centers on the frame, because they should not show in last_frame)
        if config.get('Video Export', 'circles_in_video') != "Yes":
            last_frame = frame.copy()

        # Draw all identified centers from current and previous frames
        for center in centerlist:
            cv2.circle(frame, center, 2, (0, 0, 255), 2)

        # Store the last frame for later recall showing the values (after drawing all centers on the frame)
        if config.get('Video Export', 'circles_in_video') == "Yes":
            last_frame = frame.copy()

        # Crop the frame for export
        cropped_frame = frame[:, min_x_videoexport:max_x_videoexport]

        # Write the frame to the video
        out.write(cropped_frame)

        # Check for how many consecutive frames no center (circle) was detected and if a new one was detected, reset the counter
        if len(centerlist) == centerlist_length:
            frames_without_center = frames_without_center + 1
        else:
            # Reset the frames with no center counter to 0
            frames_without_center = 0

        # Check if centers (circles) were detected
        if centerlist_length > 0:

            # Check if the latest center is close to the pins and if there were some defined consecutive frames without a center found. If yes, calculate the ball speed
            if centerlist[-1][1] - config.getint('Video Export', 'top_detection_bounds_margin') <= detection_bounds[0][0][1] and frames_without_center > config.getint('Calculations', 'frames_without_center') and not ball_speed:

                # Calculate the ball speed, by calculating the seconds passed and dividing 13.5 m (distance from top arrow to pins) by it and multiplying by 3.6 to convert m/s to km/h
                ball_speed = round(config.getfloat('Calculations', 'length_arrows_to_pins') / (time.time() - start_time) * 3.6, 1)

            # Check if the latest center is before the arrows, there is no ball speed calculated yet and the start time wasn't set
            if centerlist[-1][1] < min_y_arrows and not ball_speed and not start_time:

                # Define the start time of the speed measurement
                start_time = time.time()

        # Store current centerlist length for comparison with next frame
        centerlist_length = len(centerlist)


    # Release video capture
    cap.release()

    # Release the Debugging VideoWriter if enabled
    if config.get('Ball Detection', 'show_debugging_image') == "Yes":
        debug_out.release()

    # Check if we have enough center points to initiate the calculations
    if not len(centerlist) > config.getint('Calculations', 'min_centers_for_calculation'):

        # Write empty tracking data if we do not have enough centers for calculation
        tracking_data = {
            'ball_speed': "n/a",
            'position_at_foul_line': "n/a",
            'position_at_arrows': "n/a",
            'position_at_breakpoint': "n/a"
        }

        with open('ball_tracking_data_lane_' + str(lane_number) +'.json', 'w') as f:
            json.dump(tracking_data, f)

        # Emit a Tracking failed signal
        signal_router.tracking_unsuccessful.emit()
        return None


    ################
    # Calculations #
    ################

    # Note: Some of these calculations are a not to easily understandable mathematical processes. See documentation for explanation (does not exist yet)

    ## ~~ Calculate a fitted line through all center points ~~ ##

    # Convert centerlist to numpy array for easier manipulation to fit a curve through all centers
    center_array = np.array(centerlist)

    # Divide center_array into equal subsets for seperate fitting
    number_of_subsets_for_fitting = config.getint('Calculations', 'fitting_subsets')
    subset_size = len(center_array) // number_of_subsets_for_fitting
    subsets = [center_array[i:i + subset_size] for i in range(0, len(center_array), subset_size)]

    # Define a variable to store all fitted points from all three curves
    complete_fitted_curve_points = []

    # Fit a polynomial to each subset
    for i in range(len(subsets)):
        subset = subsets[i]
        x_subset = subset[:, 0]
        y_subset = subset[:, 1]
        coefficients = np.polyfit(x_subset, y_subset, 2) # Fit a quadratic fit to the subset
        fitted_curve = np.poly1d(coefficients)

        # Generate points for the fitted curve
        x_fit_subset = np.linspace(min(x_subset), max(x_subset), config.getint('Calculations', 'amount_of_points'))
        y_fit_subset = fitted_curve(x_fit_subset)

        # Combine the points from all the fitted curves
        points = np.column_stack((x_fit_subset, y_fit_subset))
        complete_fitted_curve_points.extend(points)

    # Convert the combined points to a numpy array
    complete_fitted_curve_points = np.array(complete_fitted_curve_points)

    # Sort all points for decreasing (vertical) y-value to allow correct drawing of the curve
    sorted_points = complete_fitted_curve_points[np.argsort(complete_fitted_curve_points[:, 1])[::-1]]

    # Smooth the combined curve
    smoothed_curve = SmoothCurve(sorted_points)

    ## ~~ Calculate the position at the arrows ~~ ##

    # Find the fitted point from SmoothCurve closed to the mean_y_arrows by calculating the distance from each point to mean_y_arrows and getting the minimal distance
    distances_to_arrows = [abs(point[1] - mean_y_arrows) for point in smoothed_curve]
    arrow_coordinate = smoothed_curve[np.argmin(distances_to_arrows)]

    # Calculate the board at the arrows, round it to half numbers.
    ball_pos_at_arrows=round(((right_bound_at_arrows[0] - arrow_coordinate[0])/(right_bound_at_arrows[0]-left_bound_at_arrows[0])*39) * 2) / 2 # There are 39 boards on a bowling lane

    ## ~~ Calculate the position at the foul line through interpolation ~~ ##

    # Calculate the lay_down board at the foul line by starting to define how many points before the mean_y_arrows are used for linear regression. All should be used excluding the first points as defined in settings
    number_of_points_for_regression = np.argmin(distances_to_arrows) - config.getint('Calculations', 'foulline_excluded_points')

    # Define the array of points used and convert to numpy array
    points_used_for_regression = np.array(smoothed_curve[np.argmin(distances_to_arrows) - number_of_points_for_regression:np.argmin(distances_to_arrows)])

    # Fit a line (straight regression) through the points
    vx_regres, vy_regres, x_regres, y_regres = cv2.fitLine(points_used_for_regression, cv2.DIST_L2, 0, 0.01, 0.01)

    # Define the equation for the fitted line
    m_regres = vy_regres / vx_regres
    c_regres = y_regres - m_regres * x_regres

    # Calculate the intersection point of the fitted line to the horizontal line at the tip of fourth arrow and at the tip of first arrow
    x_intersect_4th_arrow = float((min_y_arrows - c_regres[0])/m_regres[0])
    x_intersect_1st_arrow = float((max_y_arrows - c_regres[0])/m_regres[0])
    intersection_point_4th_arrow = (round(x_intersect_4th_arrow), min_y_arrows)
    intersection_point_1st_arrow = (round(x_intersect_1st_arrow), max_y_arrows)

    # Calculate the lane bounds and the distance between left and right bound at the fourth arrow (min_y_arrows) and first arrow (max_y_arrows)
    left_bound_at_min_y_arrows = InterpolatePoint(lane_bounds[0][0], lane_bounds[0][3], min_y_arrows)
    right_bound_at_min_y_arrows = InterpolatePoint(lane_bounds[0][1], lane_bounds[0][2], min_y_arrows)
    left_bound_at_max_y_arrows = InterpolatePoint(lane_bounds[0][0], lane_bounds[0][3], max_y_arrows)
    right_bound_at_max_y_arrows = InterpolatePoint(lane_bounds[0][1], lane_bounds[0][2], max_y_arrows)
    bound_difference_at_min_y_arrows = right_bound_at_min_y_arrows[0] - left_bound_at_min_y_arrows[0]
    bound_difference_at_max_y_arrows = right_bound_at_max_y_arrows[0] - left_bound_at_max_y_arrows[0]

    # Calculate the distance from the right lane bound (gutter) in boards at the top of the fourth and first arrow respectively
    gutter_distance_4th_arrow = (right_bound_at_min_y_arrows[0] - x_intersect_4th_arrow) / bound_difference_at_min_y_arrows * 39
    gutter_distance_1st_arrow = (right_bound_at_max_y_arrows[0] - x_intersect_1st_arrow) / bound_difference_at_max_y_arrows * 39

    # Assuming the straight line, take the board difference between the tips of the 1st and 4th arrow (45 cm length, defined as minmax_arrow_distance) and extrapolate it to the 480 cm distance between the top of the 4th arrow and the foul line (defined as foulline_arrow_distance)
    board_lay_down = round(gutter_distance_4th_arrow + (gutter_distance_1st_arrow - gutter_distance_4th_arrow) / config.getint('Calculations', 'minmax_arrow_distance') * config.getint('Calculations', 'foulline_arrow_distance')) * 2 / 2

    ## ~~ Calculate the position at the breakpoint ~~ ##

    # Define the minimal distance from the gutter as infinity (such that any distance will be smaller)
    min_distance_from_gutter = float('inf')

    # Define the array for storing distances to the gutter for all points of the smoothed_curve
    distances_to_gutter = []

    # Calculate the distance to the gutter (in boards) for each point from smoothed_curve and store it in the distances_to_gutter array
    for point in smoothed_curve:
        # Intersection point with the right lane bounds at the height of the point
        right_bound_at_point = InterpolatePoint(lane_bounds[0][1], lane_bounds[0][2], point[1])

        # Intersection point with the left lane bounds at the height of the point
        left_bound_at_point = InterpolatePoint(lane_bounds[0][0], lane_bounds[0][3], point[1])

        # Calculate the distance in boards (rounded to half a board) of each point to the right gutter
        distance_to_right_gutter = round(((right_bound_at_point[0] - point [0])/(right_bound_at_point[0] - left_bound_at_point[0]) * 39) * 2) / 2

        # Append each distance to the distances_to_gutter list
        distances_to_gutter.append(distance_to_right_gutter)

    # Check for every distance to the gutter if it is smaller than the current minimal and if yes, define it as the new minimal
    for i, point in enumerate(smoothed_curve):
        if distances_to_gutter[i] < min_distance_from_gutter:
            min_distance_from_gutter = distances_to_gutter[i]
            point_closest_to_gutter = smoothed_curve[i]

    #############################
    # Save calculations to file #
    #############################

    # Save the ball_speed as "n/a" if it wasn't calculated
    if ball_speed is None:
        ball_speed = "n/a"

    # If the minimal distance from the gutter is slightly negative, round it to 0.
    if 0 > min_distance_from_gutter >= -0.5:
        min_distance_from_gutter = 0.0

    # If the minimal distance from the gutter is significantly negative, define it as a gutter
    if min_distance_from_gutter < -0.5:
        min_distance_from_gutter = "Gutter"

    # Write the tracking data
    tracking_data = {
        'ball_speed': ball_speed,
        'position_at_foul_line': board_lay_down,
        'position_at_arrows': ball_pos_at_arrows,
        'position_at_breakpoint': min_distance_from_gutter
    }

    with open('ball_tracking_data_lane_' + str(lane_number) +'.json', 'w') as f:
        json.dump(tracking_data, f)

    #####################################
    # Draw calculations into last_frame #
    #####################################

    # Draw the fitted, smoothed curve
    for i in range(len(smoothed_curve) - 1):
        cv2.line(last_frame, tuple(smoothed_curve[i]), tuple(smoothed_curve[i + 1]), (0, 0, 255), config.getint('Video Export', 'thickness_curve'))

    # Highlight the coordinate at the arrows and label it
    cv2.circle(last_frame, (arrow_coordinate[0],mean_y_arrows), 4, (0, 0, 0), config.getint('Video Export', 'thickness_arrow'))
    cv2.putText(last_frame, str(ball_pos_at_arrows), (arrow_coordinate[0] - 20, mean_y_arrows - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)

    # Highlight the coordinate at the breakpoint and label it
    cv2.circle(last_frame, point_closest_to_gutter, 3, (0, 0, 0), config.getint('Video Export', 'thickness_breakpoint'))
    cv2.putText(last_frame, str(min_distance_from_gutter), (point_closest_to_gutter[0] - 20, point_closest_to_gutter[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display the calculated position at the foul line
    cv2.putText(last_frame, "Foul Line: " + str(board_lay_down), (1260 , 1400), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Display the calculated ball speed
    cv2.putText(last_frame, "Ball Speed (km/h): " + str(ball_speed), (1200 , 1360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Show the lines of the top and bottom of the arrows used to calculate the board_lay_down (for debugging, if enabled)
    if config.get('Calculations', 'visualize_minmax_arrow') == "Yes":
        cv2.line(last_frame, left_bound_at_min_y_arrows, right_bound_at_min_y_arrows, (160, 160, 160), 2)
        cv2.line(last_frame, left_bound_at_max_y_arrows, right_bound_at_max_y_arrows, (160, 160, 160), 2)

    # Crop the frame for export
    cropped_last_frame = last_frame[:, min_x_videoexport:max_x_videoexport]

    # Write the static image as the last frame with calculated values
    cv2.imwrite('videos/tracked_new_' + str(lane_number) + '.png', cropped_last_frame)

    # Write the last frame multiple times for 1 second at the end of the video
    for _ in range(fps * 1):
        out.write(cropped_last_frame)

    # Release the video writer
    out.release()

    ##########################################
    # Generate transparent frame for overlay #
    ##########################################

    # Get the height and width of the last frame
    height, width, _ = last_frame.shape

    # Create a transparent image with an alpha channel
    transparent_frame = np.zeros((height, width, 4), dtype=np.uint8)
    opacity = 100

    # Draw the fitted line, position at the arrows and breakpoint from the last frame on the transparent frame
    for i in range(len(smoothed_curve) - 1):
        cv2.line(transparent_frame, tuple(smoothed_curve[i]), tuple(smoothed_curve[i + 1]), (255, 0, 0, opacity), config.getint('Video Export', 'thickness_curve'))

    cv2.circle(transparent_frame, point_closest_to_gutter, 3, (255, 255, 255, opacity), config.getint('Video Export', 'thickness_breakpoint'))
    cv2.circle(transparent_frame, (arrow_coordinate[0],mean_y_arrows), 3, (255, 255, 255, opacity), config.getint('Video Export', 'thickness_arrow'))

    # Crop the transparent frame for export and export it
    cropped_transparent_frame = transparent_frame[:, min_x_videoexport:max_x_videoexport]
    cv2.imwrite('videos/tracked_new_transparent_' + str(lane_number) + '.png', cropped_transparent_frame)

    # Emit a signal that the tracker is finished and new tracking data is available
    signal_router.tracking_data_available.emit()

