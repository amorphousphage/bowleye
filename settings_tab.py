# import used packages
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTabWidget, QGridLayout, QComboBox, QSpinBox, QDoubleSpinBox, QSpacerItem, QSizePolicy, QDesktopWidget, QToolTip, QDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QObject, QPoint, pyqtSignal
import configparser
import os
import cv2
import ast
import shutil
import numpy as np

# Create a class to handle settings storing, reading and reverting to defaults
class SettingsManager:
    def __init__(self, defaults_path='defaults.cfg', settings_path=''):
        # Set the path of the settings for a lane and the default values and load the config parser
        self.defaults_path = defaults_path
        self.settings_path = settings_path
        self.config = configparser.ConfigParser()

        # Load default settings
        self.config.read(self.defaults_path)
        self.defaults = self.LoadDefaults()

        # If settings file exists, load user settings
        if os.path.exists(self.settings_path):
            self.config.read(self.settings_path)
        else:
            # Create the settings file for the lane
            with open(self.settings_path, 'x') as f:
                pass
            # Copy default settings to settings_path
            shutil.copy(self.defaults_path, self.settings_path)
            self.config.read(self.settings_path)


    def LoadDefaults(self):
        # Obtain the default values
        defaults = {}
        for section in self.config.sections():
            defaults[section] = {}
            for key, value in self.config.items(section):
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                defaults[section][key] = value
        return defaults

    def SaveSettings(self, settings):
        # Obtain the settings to be saved
        for section, values in settings.items():
            if not self.config.has_section(section):
                self.config.add_section(section)
            for key, value in values.items():
                self.config.set(section, key, str(value))

        # Write to settings file
        with open(self.settings_path, 'w') as configfile:
            self.config.write(configfile)

    def GetSettings(self):
        # Obtain the settings
        settings = {}
        for section, defaults in self.defaults.items():
            settings[section] = {}
            for key, default_value in defaults.items():
                value = self.config.get(section, key, fallback=default_value)
                if isinstance(default_value, int):
                    settings[section][key] = int(value)
                elif isinstance(default_value, float):
                    settings[section][key] = float(value)
                else:
                    settings[section][key] = value
        return settings

    def RevertToDefaults(self):
        # Save the settings from the defaults file
        self.SaveSettings(self.defaults)

# If no lanes.cfg file is found, this dialog will be called to define the lane numbers in use
class LaneInputDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Set the window title
        self.setWindowTitle("Enter Lane Numbers")

        layout = QVBoxLayout()

        # Set an instruction, a text field and an OK button to save the file
        self.label = QLabel("Enter lane numbers separated by commas:")
        self.lineEdit = QLineEdit()
        self.okButton = QPushButton("OK")
        self.okButton.clicked.connect(self.accept)

        layout.addWidget(self.label)
        layout.addWidget(self.lineEdit)
        layout.addWidget(self.okButton)

        self.setLayout(layout)

# If the position of the pins need to be determined, the following window will be generated
class PinSelectionWindow(QWidget):
    pin_coordinates_signal = pyqtSignal(dict)

    def __init__(self, frame, lane_number):
        super().__init__()
        # Define the variables used
        self.frame = frame
        self.selected_pin = None

        # Obtain the settings
        self.settings_manager = SettingsManager(defaults_path='defaults.cfg', settings_path=f'settings_lane_{lane_number}.cfg')
        settings = self.settings_manager.GetSettings()
        pin_coordinates_str = settings['Pin Scorer'].get('pin_coordinates')
        # Convert the pin coordinates to a python array
        if isinstance(pin_coordinates_str, str):
            self.pin_coordinates = ast.literal_eval(pin_coordinates_str)
        else:
            self.pin_coordinates = pin_coordinates_str

        # Calculate the scaling of the frame down to 600x400 pixel for display to set the correct coordinates
        self.height_scalar = 400 / settings['Recorder'].get('pins_camera_x_resolution')
        self.width_scalar = 600 / settings['Recorder'].get('pins_camera_y_resolution')
        print (self.height_scalar, self.width_scalar)

        self.initUI()

    def initUI(self):
        # Set window properties
        self.setWindowTitle("Pin Placement Selection")
        self.setGeometry(100, 100, 800, 600)

        # Create the main layout
        main_layout = QHBoxLayout(self)

        # Left side - Pin Buttons
        button_layout = QVBoxLayout()
        for i in range(1, 11):
            button = QPushButton(f"{i}", self)
            button.clicked.connect(self.PinButtonClicked)
            button_layout.addWidget(button)
        main_layout.addLayout(button_layout)

        # Right side - Image Display
        self.rightside_layout = QVBoxLayout(self)
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignTop)
        self.rightside_layout.addWidget(self.image_label)

        # Right side - Instructions
        self.instructions_label = QLabel(self)
        self.instructions_label.setText(
            "- Please select a pin on the left and then click the pin's neckband in the image.<br>"
            "- For the 5 pin, click on the pin's head as the neck band should not be visible.<br>"
            "- Simply close this window when all pins have been marked."
        )
        self.selected_pin_label = QLabel(self)
        self.selected_pin_label.setText("Selected Pin: Currently no pin is selected.")
        self.rightside_layout.addWidget(self.instructions_label)
        self.rightside_layout.addWidget(self.selected_pin_label)

        # Add the right side layout to the main layout
        main_layout.addLayout(self.rightside_layout)

        # Make a copy of the frame without added pin coordinates
        self.empty_frame = self.frame.copy()

        # Display the frame
        self.DisplayPinFrame(self.frame)

    def DisplayPinFrame(self, frame):
        # Resize the image using OpenCV
        resized_frame = cv2.resize(frame, (600, 400), interpolation=cv2.INTER_AREA)

        # Draw all the coordinates of the pins into the frame
        for pin_number, (x, y) in self.pin_coordinates.items():
            cv2.circle(resized_frame, (round(x * self.width_scalar), round(y * self.height_scalar)), 3, (255, 0, 0), -1)
            cv2.putText(resized_frame, str(pin_number), (round(x * self.width_scalar) - 3, round(y * self.height_scalar) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 0, 0), 2)

        # Convert the resized image to QImage
        height, width, channel = resized_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(resized_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Convert QImage to QPixmap and set it on the QLabel
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap)

        # Enable mouse click event on the image
        self.image_label.mousePressEvent = self.ImageClicked

    def PinButtonClicked(self):
        sender = self.sender()
        self.selected_pin = sender.text()
        self.selected_pin_label.setText(f"Selected Pin: Pin No. {self.selected_pin} is currently selected.")

    def ImageClicked(self, event):
        if self.selected_pin:
            self.empty_frame_copy = self.empty_frame.copy()
            pos = event.pos()
            # Store the selected coordinate for the pin
            self.pin_coordinates[self.selected_pin] = (round(pos.x() / self.width_scalar), round(pos.y() / self.height_scalar))
            print(f"Pin {self.selected_pin} selected at {round(pos.x() / self.width_scalar)}, {round(pos.y() / self.height_scalar)}")
            self.DisplayPinFrame(self.empty_frame_copy)
        else:
            QMessageBox.warning(None, "No pin was selected.", "First select a pin on the left and subsequently click on the pins neck band in the image (or the pin head for the 5 pin)")

    def closeEvent(self, event):
        # Emit the signal with the pin coordinates when the window is closed
        self.pin_coordinates_signal.emit(self.pin_coordinates)
        event.accept()  # Accept the event to close the window

# Create a class, that displays a question mark besides the label and once clicked, displays a tool tip
class HelpIconLabel(QLabel):
    def __init__(self, tooltip_text, parent=None):
        super().__init__(parent)
        self.setPixmap(QPixmap("templates/question_mark.png").scaled(30, 30))
        self.tooltip_text = tooltip_text
        self.tooltip_label = QLabel(self)
        self.tooltip_label.setWindowFlags(Qt.ToolTip)
        self.tooltip_label.setStyleSheet("border: 1px solid black;")
        self.tooltip_label.hide()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hideToolTip)

    # Define a function that displays the tooltip for a set duration when the question mark is clicked
    def mousePressEvent(self, event):
        self.showToolTip(event.globalPos())
        self.timer.start(6000)  # Display for 6 seconds

    # Define a function to show the tooltip
    def showToolTip(self, pos):
        self.tooltip_label.setText(self.tooltip_text)
        self.tooltip_label.adjustSize()
        self.tooltip_label.move(pos)
        self.tooltip_label.show()

    # Define a function to hide the tooltip
    def hideToolTip(self):
        self.tooltip_label.hide()

# Create the class for drawing the Settings Tab
class SettingsTab(QWidget):

    def __init__(self):
        super().__init__()
        # Define a variable to set whether lane numbers could be loaded or not
        lanes_successfully_loaded = False

        # If the lane numbers could not be loaded, check if the lanes.cfg exists. If not, call the input dialog to set the lanes
        while lanes_successfully_loaded == False:
            if not os.path.exists('lanes.cfg'):
                self.OpenLaneInputDialog()
            else:
                lanes_successfully_loaded = True

        # Generate a main widget for the settings tab
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create the combobox for selecting the lane number
        self.lane_select_layout = QHBoxLayout()
        self.lane_label = QLabel("Lane: ")
        self.lane_combobox = QComboBox()
        self.lane_edit_button = QPushButton("Edit Lanes")
        self.lane_edit_button.clicked.connect(self.UpdateLanes)
        self.lane_select_layout.addWidget(self.lane_label, 0)
        self.lane_select_layout.addWidget(self.lane_combobox, 0)
        self.lane_select_layout.addWidget(self.lane_edit_button)
        self.lane_select_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        main_layout.addLayout(self.lane_select_layout)

        # Add the defined lanes to the combobox
        with open('lanes.cfg', 'r') as file:
            self.lanes = file.read().splitlines()
            self.lane_combobox.addItems(self.lanes)

        # Load the settings for the lane selected once it is selected in the combobox
        self.lane_combobox.currentIndexChanged.connect(self.LoadSettingsForSelectedLane)

        # Create Combobox and button for cloning settings
        self.lane_clone_layout = QHBoxLayout()
        self.lane_clone_label = QLabel("Select a lane to clone settings from: ")
        self.lane_clone_combobox = QComboBox()
        self.lane_clone_button = QPushButton("Clone Settings")
        self.lane_clone_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.lane_clone_layout.addWidget(self.lane_clone_label, 0)
        self.lane_clone_layout.addWidget(self.lane_clone_combobox, 0)
        self.lane_clone_layout.addWidget(self.lane_clone_button, 0)
        self.lane_clone_layout.addItem(self.lane_clone_spacer)
        main_layout.addLayout(self.lane_clone_layout)

        # Connect the cloning function to the combobox
        self.lane_clone_button.clicked.connect(self.CloneSettings)

        # Create the tab widget for the different settings_categories
        self.settingsTabWidget = QTabWidget()
        main_layout.addWidget(self.settingsTabWidget)

        # Load the settings for the initial lane selected (and update items for the cloning combobox)
        self.LoadSettingsForSelectedLane()

        # Define the Save Button
        self.save_button = QPushButton("Save Settings", self)
        self.save_button.clicked.connect(self.SaveSettings)

        # Define the Restore Defaults Button
        self.default_button = QPushButton("Restore Default Settings", self)
        self.default_button.clicked.connect(self.DefaultSettings)

        # Add the save/default buttons to the Settings Tab
        buttons_layout = QHBoxLayout()
        buttons_layout.addWidget(self.default_button)
        buttons_layout.addWidget(self.save_button)
        main_layout.addLayout(buttons_layout)

        # Set the Layout
        self.setLayout(main_layout)

    # Define a function to copy paste settings from one lane onto another
    def CloneSettings(self):
        source_path = f'settings_lane_{self.lane_clone_combobox.currentText()}.cfg'
        destination_path = f'settings_lane_{self.lane_combobox.currentText()}.cfg'
        # Check if the lane selected as a source has a settings file. If not, ask the user to save the settings for the lane in question first
        if os.path.exists(source_path) and os.path.exists(destination_path):
            shutil.copy(source_path, destination_path)
            # Check if the arrow template for a lane exists and copy it. If it doesn't exist, ask the user to set one
            if os.path.exists(f'templates/arrows_template_lane_{self.lane_clone_combobox.currentText()}.png'):
                shutil.copy(f'templates/arrows_template_lane_{self.lane_clone_combobox.currentText()}.png', f'templates/arrows_template_lane_{self.lane_combobox.currentText()}.png')
                self.LoadSettingsForSelectedLane()
            else:
                QMessageBox.critical(None, "No Arrow Template found!", "The lane from which you are trying to clone does not have an arrow template set. Please first set an arrow template for that lane.")
        else:
            QMessageBox.critical(None, "Source Lane does not have settings saved", "The lane from which you are trying to clone does not have settings saved yet. Please firstly select the lane in the settings to generate the settings file and then retry.")

    # Define a function to open the LaneInputDialog to set the lanes, and write the inputted lanes into the lanes.cfg file, one lane per line
    def OpenLaneInputDialog(self):
        lane_input_dialog = LaneInputDialog()
        if lane_input_dialog.exec_():
            lanes = lane_input_dialog.lineEdit.text().split(',')
            with open('lanes.cfg', 'w') as file:
                file.write('\n'.join(lanes))

    # Define a function to let the user know the program needs to be restarted upon changing the lanes
    def UpdateLanes(self):
        self.OpenLaneInputDialog()
        QMessageBox.information(None,"Please restart the program", "Please restart the program for the changes to take effect.")

    # Define a function to load the settings for the selected lane
    def LoadSettingsForSelectedLane(self):
        lane_number = self.lane_combobox.currentText()
        lane_settings_path = f'settings_lane_{lane_number}.cfg'
        self.settings_manager = SettingsManager(defaults_path='defaults.cfg',settings_path=lane_settings_path)
        self.LoadSettingsTabs()

        # Update the cloning settings combobox items
        self.lane_clone_combobox.clear()
        cloning_lanes = []
        for lane in self.lanes:
            if lane != self.lane_combobox.currentText():
                cloning_lanes.append(lane)
        self.lane_clone_combobox.addItems(cloning_lanes)

    # Define a function to draw all settings for a selected lane into the window
    def LoadSettingsTabs(self):
        # Clear existing tabs
        self.settingsTabWidget.clear()

        # Define the categories
        settings_categories = ["Lane Setup", "Recorder", "Ball Detection", "Calculations", "Video Export", "Pin Scorer"]

        # Add the setting_categories as tabs to the widget
        for setting in settings_categories:
            self.settingsTabWidget.addTab(self.SettingCategoriesTab(setting), setting)

    # Define the function to draw the GUI elements for each of the settings categories
    def SettingCategoriesTab(self, setting):
        # Define a function to generate the label together with the tooltip
        def CreateLabelWithTip(label_text, tooltip_text):
            label_layout = QHBoxLayout()
            label = QLabel(label_text)
            help_icon = HelpIconLabel(tooltip_text)
            label_layout.addWidget(label)
            label_layout.addWidget(help_icon)
            return label_layout

        # Define a function to generate the Button together with the tooltip
        def CreateButtonWithTip(button_text, tooltip_text, function):
            button_layout = QHBoxLayout()
            button = QPushButton(button_text)
            button.clicked.connect(function)
            help_icon = HelpIconLabel(tooltip_text)
            button_layout.addWidget(button)
            button_layout.addWidget(help_icon)
            return button_layout

        tab_widget = QWidget()

        # Define the layout to be a Grid Layout (column 0 for the label, column 1 for the data value)
        tab_layout = QGridLayout()
        # Define a distance for the spacers used
        spacer_distance = 30

        # Load the current settings
        settings = self.settings_manager.GetSettings()

        # Set the GUI elements for the tab "Lane Setup"
        if setting == "Lane Setup":
            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 0, 0)

            self.detection_bounds_button = CreateButtonWithTip("Define Detection Bounds", "Draw the area where the software should detect the ball", lambda: self.DefineBounds("detection_bounds"))
            self.detection_bounds_str = settings['Lane Setup'].get('detection_bounds')
            tab_layout.addLayout(self.detection_bounds_button, 1, 0)

            self.lane_bounds_button = CreateButtonWithTip("Define Lane Bounds", "Draw the area where the lane meets the gutter", lambda: self.DefineBounds("lane_bounds"))
            self.lane_bounds_str = settings['Lane Setup'].get('lane_bounds')
            tab_layout.addLayout(self.lane_bounds_button, 2, 0)

            self.arrow_template_button = CreateButtonWithTip("Set Arrow Template","Draw the area where the 4th arrow is to use it as a template for detection",self.DefineArrowTemplate)
            tab_layout.addLayout(self.arrow_template_button, 3, 0)

            self.arrow_threshold_label = CreateLabelWithTip("Arrow Detection Threshold", "Define the Threshold to detect the arrow template. Should be between 0 and 1")
            self.arrow_threshold_item = QDoubleSpinBox()
            self.arrow_threshold_item.setDecimals(2)
            self.arrow_threshold_item.setMinimum(0)
            self.arrow_threshold_item.setMaximum(1.0)
            self.arrow_threshold_item.setSingleStep(0.01)
            self.arrow_threshold_item.setValue(settings['Lane Setup'].get('arrow_threshold'))
            tab_layout.addLayout(self.arrow_threshold_label, 4, 0)
            tab_layout.addWidget(self.arrow_threshold_item, 4, 1)

        elif setting == "Recorder":
            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 0, 0)

            self.camera_paths_title = QLabel("<b> Camera settings </b>")
            tab_layout.addWidget(self.camera_paths_title, 1, 0)

            self.tracking_camera_path_label = CreateLabelWithTip("Select Ball Tracking Camera", "Select the path to the ball tracking camera")
            self.tracking_camera_path_item = QComboBox()
            self.tracking_camera_test_item = CreateButtonWithTip("Test Tracking Camera", "Display an image of the selected camera if possible", lambda: self.TestCameraImage("Tracking Camera"))
            tab_layout.addLayout(self.tracking_camera_path_label, 2, 0)
            tab_layout.addWidget(self.tracking_camera_path_item, 2, 1)
            tab_layout.addLayout(self.tracking_camera_test_item, 2, 2)
            # Get a list of /dev/video* devices
            video_devices = [dev for dev in os.listdir('/dev') if dev.startswith('video')]
            video_devices = sorted(video_devices, key=lambda x: int(x.replace('video', '')))
            # Add devices to the combobox
            for device in video_devices:
                self.tracking_camera_path_item.addItem(f"/dev/{device}")
            self.tracking_camera_path_item.setCurrentText(settings['Recorder'].get('tracking_camera_path'))

            self.tracking_camera_resolution_label = CreateLabelWithTip("Ball Tracking Camera Resolution", "Enter the resolution of the Ball Tracking Camera in pixel")
            self.tracking_camera_x_resolution_item = QLineEdit()
            self.tracking_camera_x_resolution_item.setText(str(settings['Recorder'].get('tracking_camera_x_resolution')))
            self.tracking_camera_y_resolution_item = QLineEdit()
            self.tracking_camera_y_resolution_item.setText(str(settings['Recorder'].get('tracking_camera_y_resolution')))
            tab_layout.addLayout(self.tracking_camera_resolution_label, 3, 0)
            tab_layout.addWidget(self.tracking_camera_x_resolution_item, 3, 1)
            tab_layout.addWidget(QLabel("x"), 3, 2)
            tab_layout.addWidget(self.tracking_camera_y_resolution_item, 3, 3)

            self.pins_camera_path_label = CreateLabelWithTip("Select Pin Camera", "Select the path to the camera in front of the pins")
            self.pins_camera_path_item = QComboBox()
            self.pins_camera_test_item = CreateButtonWithTip("Test Pins Camera",
                                                                 "Display an image of the selected camera if possible",
                                                                 lambda: self.TestCameraImage("Pins Camera"))
            tab_layout.addLayout(self.pins_camera_path_label, 4, 0)
            tab_layout.addWidget(self.pins_camera_path_item, 4, 1)
            tab_layout.addLayout(self.pins_camera_test_item, 4, 2)
            for device in video_devices:
                self.pins_camera_path_item.addItem(f"/dev/{device}")

            self.pins_camera_path_item.setCurrentText(settings['Recorder'].get('pins_camera_path'))

            self.pins_camera_resolution_label = CreateLabelWithTip("Pin Camera Resolution", "Enter the resolution of the Pin Camera in pixel")
            self.pins_camera_x_resolution_item = QLineEdit()
            self.pins_camera_x_resolution_item.setText(str(settings['Recorder'].get('pins_camera_x_resolution')))
            self.pins_camera_y_resolution_item = QLineEdit()
            self.pins_camera_y_resolution_item.setText(str(settings['Recorder'].get('pins_camera_y_resolution')))
            tab_layout.addLayout(self.pins_camera_resolution_label, 5, 0)
            tab_layout.addWidget(self.pins_camera_x_resolution_item, 5, 1)
            tab_layout.addWidget(QLabel("x"), 5, 2)
            tab_layout.addWidget(self.pins_camera_y_resolution_item, 5, 3)

            self.pins_flipped_label = CreateLabelWithTip("Flip the Pins Camera Image", "Activate this option when the pins camera is displayed upside down")
            self.pins_flipped_item = QComboBox()
            self.pins_flipped_item.addItems(["Yes","No"])
            self.pins_flipped_item.setCurrentText(str(settings['Recorder'].get('pins_flipped')))
            tab_layout.addLayout(self.pins_flipped_label, 6, 0)
            tab_layout.addWidget(self.pins_flipped_item, 6, 1)


            self.recorder_frame_rate_label = CreateLabelWithTip("Tracking Camera Frame Rate", "Set the frame rate with which the recorder should check for the ball detection")
            tab_layout.addLayout(self.recorder_frame_rate_label, 7, 0)
            self.recorder_frame_rate_item = QSpinBox()
            self.recorder_frame_rate_item.setMinimum(5)
            self.recorder_frame_rate_item.setMaximum(25)
            self.recorder_frame_rate_item.setValue(settings['Recorder'].get('recorder_frame_rate'))
            tab_layout.addWidget(self.recorder_frame_rate_item, 7, 1)

            self.pins_frame_rate_label = CreateLabelWithTip("Pin Camera Frame Rate", "Set the frame rate with which the recorder should run the camera in front of the pins")
            tab_layout.addLayout(self.pins_frame_rate_label, 8, 0)
            self.pins_frame_rate_item = QSpinBox()
            self.pins_frame_rate_item.setMinimum(25)
            self.pins_frame_rate_item.setMaximum(60)
            self.pins_frame_rate_item.setValue(settings['Recorder'].get('pins_frame_rate'))
            tab_layout.addWidget(self.pins_frame_rate_item, 8, 1)

            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 9, 1)

            self.recorder_settings_title = QLabel("<b> Recorder Software settings </b>")
            tab_layout.addWidget(self.recorder_settings_title, 10, 0)

            self.recorder_detection_bounds_start_button = CreateButtonWithTip("Select Detection Bounds to Start Recording", "Draw the area where the software should detect the ball to start the recording", lambda: self.DefineBounds("recorder_start_bounds"))
            self.recorder_start_bounds_str = settings['Recorder'].get('recorder_start_bounds')
            tab_layout.addLayout(self.recorder_detection_bounds_start_button, 11, 0)

            self.recorder_detection_bounds_end_button = CreateButtonWithTip("Select Detection Bounds to End Recording", "Draw the area where the software should detect the ball to trigger the finalizing of the recording", lambda: self.DefineBounds("recorder_end_bounds"))
            self.recorder_end_bounds_str = settings['Recorder'].get('recorder_end_bounds')
            tab_layout.addLayout(self.recorder_detection_bounds_end_button, 12, 0)

            self.export_video_buffer_label = CreateLabelWithTip("Export Video Buffer", "Set the max amount of seconds of exported to a video. This is the maximal length of an exported video")
            self.export_video_buffer_item = QSpinBox()
            self.export_video_buffer_item.setMinimum(5)
            self.export_video_buffer_item.setMaximum(30)
            self.export_video_buffer_item.setValue(settings['Recorder'].get('export_video_buffer'))
            tab_layout.addLayout(self.export_video_buffer_label, 13, 0)
            tab_layout.addWidget(self.export_video_buffer_item, 13, 1)

            self.reference_frame_distance_label = CreateLabelWithTip("Reference Frame Distance", "The software uses a reference frame (without the ball) for detection. Select how many frames before the first detected ball this frame should be recorded")
            self.reference_frame_distance_item = QSpinBox()
            self.reference_frame_distance_item.setMinimum(5)
            self.reference_frame_distance_item.setMaximum(60)
            self.reference_frame_distance_item.setValue(settings['Recorder'].get('reference_frame_distance'))
            tab_layout.addLayout(self.reference_frame_distance_label, 14, 0)
            tab_layout.addWidget(self.reference_frame_distance_item, 14, 1)

            self.time_before_detection_label = CreateLabelWithTip("Time Recorded Before Shot", "Define how many seconds should be in the final tracking video before the ball is first detected")
            self.time_before_detection_item = QDoubleSpinBox()
            self.time_before_detection_item.setMinimum(0)
            self.time_before_detection_item.setMaximum(2.5)
            self.time_before_detection_item.setSingleStep(0.1)
            self.time_before_detection_item.setDecimals(1)
            self.time_before_detection_item.setValue(settings['Recorder'].get('time_before_detection'))
            tab_layout.addLayout(self.time_before_detection_label, 15, 0)
            tab_layout.addWidget(self.time_before_detection_item, 15, 1)

            self.time_after_shot_label = CreateLabelWithTip("Time Recorded After Shot", "Define how many seconds should be in the videos after the ball enters the pins")
            self.time_after_shot_item = QDoubleSpinBox()
            self.time_after_shot_item.setMinimum(1)
            self.time_after_shot_item.setMaximum(5)
            self.time_after_shot_item.setSingleStep(0.1)
            self.time_after_shot_item.setDecimals(1)
            self.time_after_shot_item.setValue(settings['Recorder'].get('time_after_shot'))
            tab_layout.addLayout(self.time_after_shot_label, 16, 0)
            tab_layout.addWidget(self.time_after_shot_item, 16, 1)

            self.detection_threshold_label = CreateLabelWithTip("Detection Threshold", "Define how many circle must be detected to start/finish recording within each defined Detection Bounds")
            self.detection_threshold_item = QSpinBox()
            self.detection_threshold_item.setMinimum(1)
            self.detection_threshold_item.setMaximum(10)
            self.detection_threshold_item.setValue(settings['Recorder'].get('detection_threshold'))
            tab_layout.addLayout(self.detection_threshold_label, 17, 0)
            tab_layout.addWidget(self.detection_threshold_item, 17, 1)

        # Set the GUI elements for the tab "Ball Detection"
        elif setting == "Ball Detection":
            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 0, 0)

            self.image_processing_title = QLabel("<b>Image Processing Settings</b>")
            tab_layout.addWidget(self.image_processing_title, 1, 0)

            self.blurred_kernel_label = CreateLabelWithTip("Kernel Size for Blurring", "Indicate the Kernel Size for the Gaussian Blur. Higher value means more blur")
            self.blurred_kernel_item = QComboBox()
            self.blurred_kernel_item.addItems(["1", "3", "5", "7", "9"])
            self.blurred_kernel_item.setCurrentText(str(settings['Ball Detection'].get('blurred_kernel')))
            tab_layout.addLayout(self.blurred_kernel_label, 2, 0)
            tab_layout.addWidget(self.blurred_kernel_item, 2, 1)

            self.blurred_sigma_label = CreateLabelWithTip("Set Standard Deviation for Blurring", "Large Standard Deviation Values increase blurring")
            self.blurred_sigma_item = QSpinBox()
            self.blurred_sigma_item.setMinimum(0)
            self.blurred_sigma_item.setMaximum(10)
            self.blurred_sigma_item.setValue(settings['Ball Detection'].get('blurred_sigma'))
            tab_layout.addLayout(self.blurred_sigma_label, 3, 0)
            tab_layout.addWidget(self.blurred_sigma_item, 3, 1)

            self.binary_threshold_label = CreateLabelWithTip("Binary Threshold", "Lower Thresholds show more details, while higher thresholds remove details")
            self.binary_threshold_item = QSpinBox()
            self.binary_threshold_item.setMinimum(20)
            self.binary_threshold_item.setMaximum(100)
            self.binary_threshold_item.setValue(settings['Ball Detection'].get('binary_threshold'))
            tab_layout.addLayout(self.binary_threshold_label, 4, 0)
            tab_layout.addWidget(self.binary_threshold_item, 4, 1)

            self.binary_max_label = CreateLabelWithTip("Binary Maximum Value", "Maximum Value defines the brightness of the changes in the image")
            self.binary_max_item = QSpinBox()
            self.binary_max_item.setMinimum(50)
            self.binary_max_item.setMaximum(300)
            self.binary_max_item.setValue(settings['Ball Detection'].get('binary_max'))
            tab_layout.addLayout(self.binary_max_label, 5, 0)
            tab_layout.addWidget(self.binary_max_item, 5, 1)

            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 6, 0)

            self.ball_detection_title = QLabel("<b>Ball Detection Settings</b>")
            tab_layout.addWidget(self.ball_detection_title, 7, 0)

            self.min_radius_label = CreateLabelWithTip("Minimal Circle Radius", "This is the minimal radius a circle can have to be detected as the ball")
            self.min_radius_item = QSpinBox()
            self.min_radius_item.setMinimum(5)
            self.min_radius_item.setMaximum(200)
            self.min_radius_item.setValue(settings['Ball Detection'].get('min_radius'))
            tab_layout.addLayout(self.min_radius_label, 8, 0)
            tab_layout.addWidget(self.min_radius_item, 8, 1)

            self.max_radius_label = CreateLabelWithTip("Maximal Circle Radius", "This is the maximal radius a circle can have to be detected as the ball")
            self.max_radius_item = QSpinBox()
            self.max_radius_item.setMinimum(30)
            self.max_radius_item.setMaximum(500)
            self.max_radius_item.setValue(settings['Ball Detection'].get('max_radius'))
            tab_layout.addLayout(self.max_radius_label, 9, 0)
            tab_layout.addWidget(self.max_radius_item, 9, 1)

            self.max_horizontal_pixel_difference_label = CreateLabelWithTip("Maximal Horizontal Pixel Difference", "Set the maximal horizontal change in pixel for a circle detected after a previous one")
            self.max_horizontal_pixel_different_item = QSpinBox()
            self.max_horizontal_pixel_different_item.setMinimum(20)
            self.max_horizontal_pixel_different_item.setMaximum(500)
            self.max_horizontal_pixel_different_item.setValue(settings['Ball Detection'].get('max_horizontal_pixel_difference'))
            tab_layout.addLayout(self.max_horizontal_pixel_difference_label, 10, 0)
            tab_layout.addWidget(self.max_horizontal_pixel_different_item, 10, 1)

            self.max_vertical_pixel_difference_label = CreateLabelWithTip("Maximal Vertical Pixel Difference", "Set the maximal vertical change in pixel for a circle detected after a previous one")
            self.max_vertical_pixel_different_item = QSpinBox()
            self.max_vertical_pixel_different_item.setMinimum(20)
            self.max_vertical_pixel_different_item.setMaximum(500)
            self.max_vertical_pixel_different_item.setValue(settings['Ball Detection'].get('max_vertical_pixel_difference'))
            tab_layout.addLayout(self.max_vertical_pixel_difference_label, 11, 0)
            tab_layout.addWidget(self.max_vertical_pixel_different_item, 11, 1)

            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 12, 0)

            self.debug_image_title = QLabel("<b>Debugging Image</b>")
            tab_layout.addWidget(self.debug_image_title, 13, 0)

            self.toggle_debugging_image_label = CreateLabelWithTip("Save Debugging Image Video", "Save a debugging video showing the images, which the program uses for Ball detection")
            self.toggle_debugging_image_item = QComboBox()
            self.toggle_debugging_image_item.addItems(["Yes", "No"])
            self.toggle_debugging_image_item.setCurrentText(str(settings['Ball Detection'].get('show_debugging_image')))
            tab_layout.addLayout(self.toggle_debugging_image_label, 14, 0)
            tab_layout.addWidget(self.toggle_debugging_image_item, 14, 1)

            self.change_debugging_image_label = CreateLabelWithTip("Set Debugging Image Type", "Choose between the Image Showing the Difference to the Reference or the processed binary image")
            self.change_debugging_image_item = QComboBox()
            self.change_debugging_image_item.addItems(["Binary", "Difference only"])
            self.change_debugging_image_item.setCurrentText(str(settings['Ball Detection'].get('debugging_image_type')))
            tab_layout.addLayout(self.change_debugging_image_label, 15, 0)
            tab_layout.addWidget(self.change_debugging_image_item, 15, 1)

        # Set the GUI elements for the tab "Calculations"
        elif setting == "Calculations":
            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 0, 0)

            self.general_title = QLabel("<b>General</b>")
            tab_layout.addWidget(self.general_title, 1, 0)

            self.min_centers_for_calculation_label = CreateLabelWithTip("Minimum Number of Circles detected to start calculation", "Determine the minimal amount of circles to be detected, to trigger video export and all calculations")
            self.min_centers_for_calculation_item = QSpinBox()
            self.min_centers_for_calculation_item.setMinimum(2)
            self.min_centers_for_calculation_item.setMaximum(30)
            self.min_centers_for_calculation_item.setValue(settings['Calculations'].get('min_centers_for_calculation'))
            tab_layout.addLayout(self.min_centers_for_calculation_label, 2, 0)
            tab_layout.addWidget(self.min_centers_for_calculation_item, 2, 1)

            self.fitting_subsets_label = CreateLabelWithTip("Subsets for Fitting the Ball Curve", "The detected circles will be split into subsets to fit quadratic equations to it. Determine into how many subsets it should be split")
            self.fitting_subsets_item = QSpinBox()
            self.fitting_subsets_item.setMinimum(2)
            self.fitting_subsets_item.setMaximum(10)
            self.fitting_subsets_item.setValue(settings['Calculations'].get('fitting_subsets'))
            tab_layout.addLayout(self.fitting_subsets_label, 3, 0)
            tab_layout.addWidget(self.fitting_subsets_item, 3, 1)

            self.amount_of_points_label = CreateLabelWithTip("Points per Curve", "Set how many points should be calculated for the whole fitted curve")
            self.amount_of_points_item = QSpinBox()
            self.amount_of_points_item.setMinimum(50)
            self.amount_of_points_item.setMaximum(1000)
            self.amount_of_points_item.setSingleStep(50)
            self.amount_of_points_item.setValue(settings['Calculations'].get('amount_of_points'))
            tab_layout.addLayout(self.amount_of_points_label, 4, 0)
            tab_layout.addWidget(self.amount_of_points_item, 4, 1)

            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 5, 0)

            self.ball_speed_title = QLabel("<b>Ball Speed</b>")
            tab_layout.addWidget(self.ball_speed_title, 6, 0)

            self.frames_without_center_label = CreateLabelWithTip("Set Frames without Circle detected", "Define how many frames should pass without circle detection, before the ball is considered at the end of the lane. This is used for ball speed calculation")
            self.frames_without_center_item = QSpinBox()
            self.frames_without_center_item.setMinimum(2)
            self.frames_without_center_item.setMaximum(20)
            self.frames_without_center_item.setValue(settings['Calculations'].get('frames_without_center'))
            tab_layout.addLayout(self.frames_without_center_label, 7, 0)
            tab_layout.addWidget(self.frames_without_center_item, 7, 1)

            self.length_arrows_to_pins_label = CreateLabelWithTip("Length of 4th Arrow to the pins", "Set the length in meters from the 4th arrow tip to the pins")
            self.length_arrows_to_pins_item = QDoubleSpinBox()
            self.length_arrows_to_pins_item.setMinimum(10)
            self.length_arrows_to_pins_item.setMaximum(20)
            self.length_arrows_to_pins_item.setSingleStep(0.1)
            self.length_arrows_to_pins_item.setDecimals(1)
            self.length_arrows_to_pins_item.setValue(settings['Calculations'].get('length_arrows_to_pins'))
            tab_layout.addLayout(self.length_arrows_to_pins_label, 8, 0)
            tab_layout.addWidget(self.length_arrows_to_pins_item, 8, 1)

            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 9, 0)

            self.general_title = QLabel("<b>Foul Line Calculation</b>")
            tab_layout.addWidget(self.general_title, 10, 0)


            self.minmax_arrow_label = CreateLabelWithTip("Visualize min. and max. line of the arrows?", "For debugging: Show the identified maximum arrow and minimum arrow position in the video")
            self.minmax_arrow_item = QComboBox()
            self.minmax_arrow_item.addItems(["Yes", "No"])
            self.minmax_arrow_item.setCurrentText(str(settings['Calculations'].get('visualize_minmax_arrow')))
            tab_layout.addLayout(self.minmax_arrow_label, 11, 0)
            tab_layout.addWidget(self.minmax_arrow_item, 11, 1)

            self.minmax_arrow_distance_label = CreateLabelWithTip("Distance between the tip of the first to the tip of the fourth arrow in cm", "Enter the distance between the tips of the first and fourth arrow as visualized with the option above")
            self.minmax_arrow_distance_item = QSpinBox()
            self.minmax_arrow_distance_item.setMinimum(20)
            self.minmax_arrow_distance_item.setMaximum(80)
            self.minmax_arrow_distance_item.setValue(settings['Calculations'].get('minmax_arrow_distance'))
            tab_layout.addLayout(self.minmax_arrow_distance_label, 12, 0)
            tab_layout.addWidget(self.minmax_arrow_distance_item, 12, 1)

            self.foulline_arrow_distance_label = CreateLabelWithTip("Distance between the tip of the first arrow to the foul line in cm", "Enter the distance between the tip of the first arrow to the foul line")
            self.foulline_arrow_distance_item = QSpinBox()
            self.foulline_arrow_distance_item.setMinimum(300)
            self.foulline_arrow_distance_item.setMaximum(900)
            self.foulline_arrow_distance_item.setValue(settings['Calculations'].get('foulline_arrow_distance'))
            tab_layout.addLayout(self.foulline_arrow_distance_label, 13, 0)
            tab_layout.addWidget(self.foulline_arrow_distance_item, 13, 1)

            self.foulline_excluded_points_label = CreateLabelWithTip("Number of inital points to exclude from Foul Line Calculation", "Enter the number of points (starting from the foul line side) that should be excluded from the fitting of the linear equation for foul line ")
            self.foulline_excluded_points_item = QSpinBox()
            self.foulline_excluded_points_item.setMinimum(0)
            self.foulline_excluded_points_item.setMaximum(200)
            self.foulline_excluded_points_item.setValue(settings['Calculations'].get('foulline_excluded_points'))
            tab_layout.addLayout(self.foulline_excluded_points_label, 14, 0)
            tab_layout.addWidget(self.foulline_excluded_points_item, 14, 1)

        # Set the GUI elements for the tab "Video Export"
        elif setting == "Video Export":
            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 0, 0)

            self.video_properties_title = QLabel("<b>Video Properties</b>")
            tab_layout.addWidget(self.video_properties_title, 1, 0)

            self.margins_label = CreateLabelWithTip("Margins around Detection Area for Video Export", "Define how many pixels around the set Detection Bounds should be visible in the video")
            self.margins_item = QSpinBox()
            self.margins_item.setMinimum(0)
            self.margins_item.setMaximum(200)
            self.margins_item.setValue(settings['Video Export'].get('margins_video_export'))
            tab_layout.addLayout(self.margins_label, 2, 0)
            tab_layout.addWidget(self.margins_item, 2, 1)

            self.top_detection_bounds_margin_label = CreateLabelWithTip("Margin to Detection Bounds at the Pins", "Set the margin of a detected circle to the top border of the detection bounds for it to trigger Video End Trigger")
            self.top_detection_bounds_margin_item = QSpinBox()
            self.top_detection_bounds_margin_item.setMinimum(10)
            self.top_detection_bounds_margin_item.setMaximum(200)
            self.top_detection_bounds_margin_item.setValue(settings['Video Export'].get('top_detection_bounds_margin'))
            tab_layout.addLayout(self.top_detection_bounds_margin_label, 3, 0)
            tab_layout.addWidget(self.top_detection_bounds_margin_item, 3, 1)

            tab_layout.addItem(QSpacerItem(spacer_distance, spacer_distance), 4, 0)

            self.video_properties_title = QLabel("<b>Graphic Output</b>")
            tab_layout.addWidget(self.video_properties_title, 5, 0)

            self.circles_in_video_label = CreateLabelWithTip("Show the detected ball positions in the video export", "Select whether the recognized ball positions should be shown in the final video additionally to the fitted curve")
            self.circles_in_video_item = QComboBox()
            self.circles_in_video_item.addItems(["Yes", "No"])
            self.circles_in_video_item.setCurrentText(str(settings['Video Export'].get('circles_in_video')))
            tab_layout.addLayout(self.circles_in_video_label, 6, 0)
            tab_layout.addWidget(self.circles_in_video_item, 6, 1)

            self.thickness_curve_label = CreateLabelWithTip("Thickness of the fitted curve", "Set the thickness of the fitted curve in the video")
            self.thickness_curve_item = QSpinBox()
            self.thickness_curve_item.setMinimum(1)
            self.thickness_curve_item.setMaximum(8)
            self.thickness_curve_item.setValue(settings['Video Export'].get('thickness_curve'))
            tab_layout.addLayout(self.thickness_curve_label, 7, 0)
            tab_layout.addWidget(self.thickness_curve_item, 7, 1)

            self.thickness_breakpoint_label = CreateLabelWithTip("Thickness of the Breakpoint Circle","Set the thickness of the circle marking the Breakpoint")
            self.thickness_breakpoint_item = QSpinBox()
            self.thickness_breakpoint_item.setMinimum(1)
            self.thickness_breakpoint_item.setMaximum(40)
            self.thickness_breakpoint_item.setValue(settings['Video Export'].get('thickness_breakpoint'))
            tab_layout.addLayout(self.thickness_breakpoint_label, 8, 0)
            tab_layout.addWidget(self.thickness_breakpoint_item, 8, 1)

            self.thickness_arrow_label = CreateLabelWithTip("Thickness of the Arrow Circle","Set the thickness of the circle marking the Arrow Position")
            self.thickness_arrow_item = QSpinBox()
            self.thickness_arrow_item.setMinimum(1)
            self.thickness_arrow_item.setMaximum(40)
            self.thickness_arrow_item.setValue(settings['Video Export'].get('thickness_arrow'))
            tab_layout.addLayout(self.thickness_arrow_label, 9, 0)
            tab_layout.addWidget(self.thickness_arrow_item, 9, 1)

        # Set the GUI elements for the tab "Pin Scorer"
        elif setting == "Pin Scorer":

            self.time_pin_reading_after_start_label = CreateLabelWithTip("How many frames should pass before the pin score is read?",
                                                                        "Select how many frames should be recorded since the pin camera started before the score should be read")
            self.time_pin_reading_after_start_item = QSpinBox()
            self.time_pin_reading_after_start_item.setMinimum(10)
            self.time_pin_reading_after_start_item.setMaximum(200)
            self.time_pin_reading_after_start_item.setValue(settings['Pin Scorer'].get('time_pin_reading_after_start'))
            tab_layout.addLayout(self.time_pin_reading_after_start_label, 1, 0)
            tab_layout.addWidget(self.time_pin_reading_after_start_item, 1, 1)

            self.set_pin_positions_button = CreateButtonWithTip(
                "Define Pin Positions",
                "Select the location of each Pin's neck band (and for the 5-pin the head) in an image",
                self.InitializePinSelectionWindow)
            tab_layout.addLayout(self.set_pin_positions_button, 2, 0)
            self.pin_coordinates = settings['Pin Scorer'].get('pin_coordinates')

        # Add a horizontal and vertical stretch to push items to the top-left in each tab
        tab_layout.setColumnStretch(4, 1)
        tab_layout.setRowStretch(tab_layout.rowCount(),1)

        tab_widget.setLayout(tab_layout)
        return tab_widget

    # Define a function to save the settings
    def SaveSettings(self):
        new_settings = {
            'Lane Setup': {
                'arrow_threshold': self.arrow_threshold_item.value(),
                'detection_bounds': self.detection_bounds_str,
                'lane_bounds': self.lane_bounds_str
            },
            'Recorder': {
                'tracking_camera_path': self.tracking_camera_path_item.currentText(),
                'pins_camera_path': self.pins_camera_path_item.currentText(),
                'tracking_camera_x_resolution': self.tracking_camera_x_resolution_item.text(),
                'tracking_camera_y_resolution': self.tracking_camera_y_resolution_item.text(),
                'pins_camera_x_resolution': self.pins_camera_x_resolution_item.text(),
                'pins_camera_y_resolution': self.pins_camera_y_resolution_item.text(),
                'pins_flipped': self.pins_flipped_item.currentText(),
                'recorder_start_bounds': self.recorder_start_bounds_str,
                'recorder_end_bounds': self.recorder_end_bounds_str,
                'recorder_frame_rate': self.recorder_frame_rate_item.value(),
                'pins_frame_rate': self.pins_frame_rate_item.value(),
                'export_video_buffer': self.export_video_buffer_item.value(),
                'reference_frame_distance': self.reference_frame_distance_item.value(),
                'time_before_detection': self.time_before_detection_item.value(),
                'time_after_shot': self.time_after_shot_item.value(),
                'detection_threshold': self.detection_threshold_item.value()
            },
            'Ball Detection': {
                'blurred_kernel': self.blurred_kernel_item.currentText(),
                'blurred_sigma': self.blurred_sigma_item.value(),
                'binary_threshold': self.binary_threshold_item.value(),
                'binary_max': self.binary_max_item.value(),
                'min_radius': self.min_radius_item.value(),
                'max_radius': self.max_radius_item.value(),
                'max_horizontal_pixel_difference': self.max_horizontal_pixel_different_item.value(),
                'max_vertical_pixel_difference': self.max_vertical_pixel_different_item.value(),
                'show_debugging_image': self.toggle_debugging_image_item.currentText(),
                'debugging_image_type': self.change_debugging_image_item.currentText(),
            },
            'Calculations': {
                'min_centers_for_calculation': self.min_centers_for_calculation_item.value(),
                'fitting_subsets': self.fitting_subsets_item.value(),
                'amount_of_points': self.amount_of_points_item.value(),
                'frames_without_center': self.frames_without_center_item.value(),
                'length_arrows_to_pins': self.length_arrows_to_pins_item.value(),
                'visualize_minmax_arrow': self.minmax_arrow_item.currentText(),
                'minmax_arrow_distance': self.minmax_arrow_distance_item.value(),
                'foulline_arrow_distance': self.foulline_arrow_distance_item.value(),
                'foulline_excluded_points': self.foulline_excluded_points_item.value(),
            },
            'Video Export': {
                'margins_video_export': self.margins_item.value(),
                'top_detection_bounds_margin': self.top_detection_bounds_margin_item.value(),
                'circles_in_video': self.circles_in_video_item.currentText(),
                'thickness_curve': self.thickness_curve_item.value(),
                'thickness_breakpoint': self.thickness_breakpoint_item.value(),
                'thickness_arrow': self.thickness_arrow_item.value(),
            },

            'Pin Scorer': {
                'time_pin_reading_after_start': self.time_pin_reading_after_start_item.value(),
                'pin_coordinates': self.pin_coordinates
            }
        }
        self.settings_manager.SaveSettings(new_settings)

    # Define a function to revert to default settings
    def DefaultSettings(self):
        self.settings_manager.RevertToDefaults()
        # Clear all tabs
        self.settingsTabWidget.clear()
        # Reload settings into UI
        settings = self.settings_manager.GetSettings()
        settings_categories = ["Lane Setup", "Recorder", "Ball Detection", "Calculations", "Video Export", "Pin Scorer"]
        for setting in settings_categories:
            self.settingsTabWidget.addTab(self.SettingCategoriesTab(setting), setting)

    # Define a function to read a camera image for testing
    def ReadCameraImage(self, camera):
        # Obtain the settings to extract the current camera path
        settings = self.settings_manager.GetSettings()
        if camera == "Tracking Camera":
            # Set the camera capture
            cap = cv2.VideoCapture(self.tracking_camera_path_item.currentText())
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.tracking_camera_x_resolution_item.text()))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.tracking_camera_y_resolution_item.text()))

            # Try to read the camera frame
            ret, frame = cap.read()
            # Release the camera after reading
            cap.release()

            # If reading the camera failed, display the alert and return None
            if not ret:
                QMessageBox.critical(None, "Could not read tracking camera feed!","Please make sure the tracking camera for this lane is correctly selected and a resolution that is supported by the camera is entered.")
                return False

        elif camera == "Pins Camera":
            # Set the camera capture
            cap = cv2.VideoCapture(self.pins_camera_path_item.currentText())
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.pins_camera_x_resolution_item.text()))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.pins_camera_y_resolution_item.text()))

            # Try to read the camera frame
            ret, frame = cap.read()
            # Release the camera after reading
            cap.release()

            # If reading the camera failed, display the alert and return None
            if not ret:
                QMessageBox.critical(None, "Could not read pins camera feed!",
                                     "Please make sure the pins camera for this lane is correctly selected and a resolution that is supported by the camera is entered.")
                return False

            # Flip the pins image if set
            if settings['Recorder'].get('pins_flipped') == "Yes":
                frame = cv2.flip(frame, -1)

        return frame

    #Function to test if the correct camera was selected (output a sample image)
    def TestCameraImage(self, camera):

        # Read the Camera frame
        frame = self.ReadCameraImage(camera)

        # Display the frame in a window
        cv2.namedWindow("Testing " + camera, cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Testing " + camera, 1800, 900)
        cv2.imshow("Testing " + camera, frame)


    # Function to define detection boundaries by drawing a trapezoid on the first frame
    def DefineBounds(self, bounds_variable):
        frame = self.ReadCameraImage("Tracking Camera")

        if frame is None:
            # If the camera feed could not be read, exit the function early.
            return

        # Generate a copy of the frame to draw the bounds on it
        frame_copy = frame.copy()

        # Define the initial detection bounds (a trapezoid with horizontal lines)
        if bounds_variable == "detection_bounds":
            bounds = np.array(ast.literal_eval(self.detection_bounds_str), dtype=np.int32)
        elif bounds_variable == "lane_bounds":
            bounds = np.array(ast.literal_eval(self.lane_bounds_str), dtype=np.int32)
        elif bounds_variable == "recorder_start_bounds":
            bounds = np.array(ast.literal_eval(self.recorder_start_bounds_str), dtype=np.int32)
        elif bounds_variable == "recorder_end_bounds":
            bounds = np.array(ast.literal_eval(self.recorder_end_bounds_str), dtype=np.int32)

        # Function to update the detection bounds based on the moved point
        def UpdateBounds(index, new_position):
            bounds[0][index] = new_position

            # Make sure the top and bottom lines remain horizontal
            if index in [0, 2]:  # first coordinates of the lower and upper line
                # Set the y-coordinate of the point with the next higher index number to match the y-coordinate of the previous point
                bounds[0][index + 1][1] = new_position[1]
            elif index in [1, 3]:  # second coordinates of the lower and upper line
                # Set the y-coordinate of the point with the next lower index number to match the y-coordinate of the current point
                bounds[0][index - 1][1] = new_position[1]

        # Draw the initial trapezoid on the frame
        if bounds_variable == "detection_bounds":
            cv2.polylines(frame_copy, [bounds], True, (255, 0, 0), 2)
            # Draw circles at the corner points
            for point in bounds[0]:
                cv2.circle(frame_copy, tuple(point), 5, (255, 0, 0), -1)
        elif bounds_variable == "lane_bounds":
            cv2.polylines(frame_copy, [bounds], True, (0, 0, 255), 1)
            # Draw circles at the corner points
            for point in bounds[0]:
                cv2.circle(frame_copy, tuple(point), 3, (0, 0, 255), -1)
        elif bounds_variable == "recorder_start_bounds" or bounds_variable == "recorder_end_bounds":
            cv2.polylines(frame_copy, [bounds], True, (0, 255, 255), 2)
            # Draw circles at the corner points
            for point in bounds[0]:
                cv2.circle(frame_copy, tuple(point), 5, (0, 255, 255), -1)

        # Flag to indicate if dragging is in progress
        dragging = False

        # Index of the point being dragged
        dragging_index = None

        # Function to handle mouse events
        def MouseCallback(event, x, y, flags, param):
            nonlocal frame_copy, dragging, dragging_index

            if event == cv2.EVENT_LBUTTONDOWN:
                # Check if the mouse click is near any of the points
                for i, point in enumerate(bounds[0]):
                    if abs(x - point[0]) < 10 and abs(y - point[1]) < 10:
                        dragging = True
                        dragging_index = i
                        break

            elif event == cv2.EVENT_MOUSEMOVE:
                if dragging:
                    # Update the position of the dragging point
                    UpdateBounds(dragging_index, (x, y))
                    frame_copy = frame.copy()
                    # Draw the initial trapezoid on the frame
                    if bounds_variable == "detection_bounds":
                        cv2.polylines(frame_copy, [bounds], True, (255, 0, 0), 2)
                        # Draw circles at the corner points
                        for point in bounds[0]:
                            cv2.circle(frame_copy, tuple(point), 5, (255, 0, 0), -1)
                        cv2.imshow("Define Detection Bounds", frame_copy)
                    elif bounds_variable == "lane_bounds":
                        cv2.polylines(frame_copy, [bounds], True, (0, 0, 255), 1)
                        # Draw circles at the corner points
                        for point in bounds[0]:
                            cv2.circle(frame_copy, tuple(point), 3, (0, 0, 255), -1)
                        cv2.imshow("Define Lane Bounds", frame_copy)
                    elif bounds_variable == "recorder_start_bounds":
                        cv2.polylines(frame_copy, [bounds], True, (0, 255, 255), 2)
                        # Draw circles at the corner points
                        for point in bounds[0]:
                            cv2.circle(frame_copy, tuple(point), 5, (0, 255, 255), -1)
                        cv2.imshow("Define Recorder Start Bounds", frame_copy)
                    elif bounds_variable == "recorder_end_bounds":
                        cv2.polylines(frame_copy, [bounds], True, (0, 255, 255), 2)
                        # Draw circles at the corner points
                        for point in bounds[0]:
                            cv2.circle(frame_copy, tuple(point), 5, (0, 255, 255), -1)
                        cv2.imshow("Define Recorder End Bounds", frame_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        # Create a window and set mouse callback function
        if bounds_variable == "detection_bounds":
            cv2.namedWindow("Define Detection Bounds", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Define Detection Bounds", 1800, 900)
            cv2.imshow("Define Detection Bounds", frame_copy)
            cv2.setMouseCallback("Define Detection Bounds", MouseCallback)

            while cv2.getWindowProperty("Define Detection Bounds", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(50)
            bounds_str = str(bounds.tolist())
            self.detection_bounds_str = bounds_str
            cv2.destroyWindow("Define Detection Bounds")

        if bounds_variable == "lane_bounds":
            cv2.namedWindow("Define Lane Bounds", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Define Lane Bounds", 1800, 900)
            cv2.imshow("Define Lane Bounds", frame_copy)
            cv2.setMouseCallback("Define Lane Bounds", MouseCallback)

            while cv2.getWindowProperty("Define Lane Bounds", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(50)
            bounds_str = str(bounds.tolist())
            self.lane_bounds_str = bounds_str
            cv2.destroyWindow("Define Lane Bounds")

        if bounds_variable == "recorder_start_bounds":
            cv2.namedWindow("Define Recorder Start Bounds", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Define Recorder Start Bounds", 1800, 900)
            cv2.imshow("Define Recorder Start Bounds", frame_copy)
            cv2.setMouseCallback("Define Recorder Start Bounds", MouseCallback)

            while cv2.getWindowProperty("Define Recorder Start Bounds", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(50)
            bounds_str = str(bounds.tolist())
            self.recorder_start_bounds_str = bounds_str
            cv2.destroyWindow("Define Recorder Start Bounds")

        if bounds_variable == "recorder_end_bounds":
            cv2.namedWindow("Define Recorder End Bounds", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Define Recorder End Bounds", 1800, 900)
            cv2.imshow("Define Recorder End Bounds", frame_copy)
            cv2.setMouseCallback("Define Recorder End Bounds", MouseCallback)

            while cv2.getWindowProperty("Define Recorder End Bounds", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.waitKey(50)
            bounds_str = str(bounds.tolist())
            self.recorder_end_bounds_str = bounds_str
            cv2.destroyWindow("Define Recorder End Bounds")

    # Function to define the arrow template
    def DefineArrowTemplate(self):
        frame = self.ReadCameraImage("Tracking Camera")

        if frame is None:
            # If the camera feed could not be read, exit the function early.
            return

        frame_copy = frame.copy()

        # Variables to store the rectangle coordinates
        rect_start = None
        rect_end = None
        drawing = False

        # Mouse callback function to draw the rectangle
        def MouseCallback(event, x, y, flags, param):
            nonlocal rect_start, rect_end, drawing, frame_copy

            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                rect_start = (x, y)
                rect_end = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    rect_end = (x, y)
                    frame_copy = frame.copy()
                    cv2.rectangle(frame_copy, rect_start, rect_end, (0, 255, 0), 2)
                    cv2.imshow("Define Arrow Template", frame_copy)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                rect_end = (x, y)
                frame_copy = frame.copy()
                cv2.rectangle(frame_copy, rect_start, rect_end, (0, 255, 0), 2)
                cv2.imshow("Define Arrow Template", frame_copy)

        # Create a window and set mouse callback function
        cv2.namedWindow("Define Arrow Template", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Define Arrow Template", 1800, 900)
        cv2.imshow("Define Arrow Template", frame_copy)
        cv2.setMouseCallback("Define Arrow Template", MouseCallback)

        # Wait until the window is closed by the user
        while cv2.getWindowProperty("Define Arrow Template", cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(50)

        # Extract and save the arrow template
        if rect_start and rect_end:
            x1, y1 = rect_start
            x2, y2 = rect_end
            arrow_template = frame[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
            cv2.imwrite(f'templates/arrows_template_lane_{self.lane_combobox.currentText()}.png', arrow_template)

        cv2.destroyWindow("Define Arrow Template")

    def InitializePinSelectionWindow(self):
        pin_frame = self.ReadCameraImage("Pins Camera")
        self.pin_selection_window = PinSelectionWindow(pin_frame, self.lane_combobox.currentText())
        self.pin_selection_window.pin_coordinates_signal.connect(self.ReceivePinCoordinates)
        self.pin_selection_window.show()


    def ReceivePinCoordinates(self, coordinates):
        # Store the pin coordinates upon closing the PinSelectionWindow
        self.pin_coordinates = coordinates
        print(self.pin_coordinates)
