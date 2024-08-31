# import used packages
import os
import datetime
import shutil
import configparser
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QStackedLayout, QTabWidget, QRadioButton, QButtonGroup, QGroupBox, QPushButton, QSpinBox, QLineEdit, QLabel, QDesktopWidget, QStyle, QComboBox, QCheckBox, QTableWidget, QTableWidgetItem, QGridLayout, QSpacerItem, QMessageBox, QSizePolicy
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSlot, QTimer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import QPixmap, QFont, QMovie
from pyqt_switch import PyQtSwitch
from tracking_data_updater import TrackingDataUpdater
from recorder import RecorderWorker
from signal_router import signal_router

class LaneTab(QWidget):

    def __init__(self, lane_number):
        super().__init__()

        self.lane_number = lane_number

        # Set up a layout for the lane tabs containing three columns
        self.tab_layout = QHBoxLayout()
        self.col1_layout = QVBoxLayout()
        self.col2_layout = QVBoxLayout()
        self.col2_stacked_layout = QStackedLayout()  # This stacked layout will be used to display the overlay over the ball_tracking_video and/or _image
        self.col3_layout = QVBoxLayout()
        self.tab_layout.addLayout(self.col1_layout)
        self.tab_layout.addLayout(self.col2_layout)
        self.tab_layout.addLayout(self.col3_layout)

        self.setLayout(self.tab_layout)

        #####################
        # Setup UI elements #
        #####################

        # Obtain the Screen Width and Height as reference for the widget placement
        screen_size = QDesktopWidget().screenGeometry()

        # Single Shot Button
        self.single_shot_button = QRadioButton("Single Shot Mode")
        # Set the Single Shot Button to be checked
        self.single_shot_button.setChecked(True)
        # Set the first Widget of column 1 to a minimum width of 30% of Screen Width
        self.single_shot_button.setMinimumWidth(round(screen_size.width() * 0.3))
        # Add the Single Shot Button to the Tab Layout (first row, first column)
        self.col1_layout.addWidget(self.single_shot_button)
        # Record Mode Button
        self.record_mode_button = QRadioButton("Record Multiple Shots Mode")
        # Add the Record Mode Button to the Tab Layout (second row, first column)
        self.col1_layout.addWidget(self.record_mode_button)
        # Create Button Group and add Single Shot and Record Mode Button
        self.Mode_buttons = QButtonGroup()
        self.Mode_buttons.addButton(self.single_shot_button)
        self.Mode_buttons.addButton(self.record_mode_button)
        # Connect the toggled signal of the radio buttons to the updateFieldVisibility function
        self.single_shot_button.toggled.connect(self.updateFieldVisibility)
        self.record_mode_button.toggled.connect(self.updateFieldVisibility)

        # Create a GroupBox to group all items for record mode to be hidden in Live View
        self.record_mode_inputs = QGroupBox()
        # Add the GroupBox to the first column Layout (third row)
        self.col1_layout.addWidget(self.record_mode_inputs)
        # Create an empty placeholder to cover the area of the record_mode_inputs when they are hidden and add it to the layout at the same place
        self.record_mode_placeholder = QGroupBox()
        # Add the Placeholder GroupBox at the same place as the record_mode_inputs
        self.col1_layout.addWidget(self.record_mode_placeholder)

        # Create a layout for all fields within the record_mode_inputs GroupBox
        self.record_mode_layout = QGridLayout(self.record_mode_inputs)
        # Align the layout content to the top
        self.record_mode_layout.setAlignment(Qt.AlignTop)
        # Create a layout for all player fields within the GroupBox
        self.player_layout = QGridLayout()
        # Align the layout content to the top
        self.player_layout.setAlignment(Qt.AlignTop)
        # Add the layout to the record_mode_layout (second row, first column, spanning three columns)
        self.record_mode_layout.addLayout(self.player_layout, 2, 0, 1, 3)

        # Create the labels for "Number of Players" and add it to the record_mode_layout (first row, first column)
        self.number_of_players_label = QLabel("Number of Players")
        self.record_mode_layout.addWidget(self.number_of_players_label, 0, 0)
        # Create spin box to select the number of players and define it to be between 1-6 players
        self.player_count_spinbox = QSpinBox()
        self.player_count_spinbox.setMinimum(1)
        self.player_count_spinbox.setMaximum(6)
        # Set the spin box to 5% of the screen width
        self.player_count_spinbox.setFixedWidth(round(screen_size.width() * 0.05))
        # Generate & Update the Player fields depending on the value of the SpinBox and also run it when booting the program
        self.updatePlayerFields()
        self.player_count_spinbox.valueChanged.connect(self.updatePlayerFields)
        # Add the SpinBox to the record_mode_layout (first row, second column)
        self.record_mode_layout.addWidget(self.player_count_spinbox, 0, 1)

        # Create the Next Player Button
        self.next_player_button = QPushButton("Next Player/Frame")
        self.record_mode_layout.addWidget(self.next_player_button, 5, 0)
        # Initially hide this button
        self.next_player_button.setVisible(False)
        # Define the variable for the currently selected player who is playing
        self.selected_player = None
        # Define the variable for the current frame of the selected player
        self.selected_player_current_frame = None
        # Define the variable for the current game of the selected player
        self.selected_player_current_game = None
        # Connect the Next_player_button to the function to obtain all player names and select the next player
        self.next_player_button.clicked.connect(self.nextPlayer)

        # Create the New Game Button
        self.new_game_button = QPushButton("Next Game")
        self.record_mode_layout.addWidget(self.new_game_button, 5, 1)
        # Initially hide this button
        self.new_game_button.setVisible(False)
        # Connect the new_game_button to the function to select the first player, increase the game counter by one and reset the frame counter
        self.new_game_button.clicked.connect(self.nextGame)

        # Create Confirm/Edit Player Button
        self.edit_player_button = QPushButton("Confirm Players")
        self.record_mode_layout.addWidget(self.edit_player_button, 5, 2)
        # Connect the new_game_button to the function to select the first player, increase the game counter by one and reset the frame counter
        self.edit_player_button.clicked.connect(self.confirmPlayers)

        # Create a layout for the recorder and its switch
        self.recorder_layout = QHBoxLayout()
        # Create the switch to enable the recorder
        self.recorder_switch_icon = QLabel()
        self.recorder_switch_icon.setPixmap(QPixmap("templates/recorder_icon.png"))
        self.recorder_switch = PyQtSwitch()
        self.recorder_switch.toggled.connect(self.ToggleRecorder)
        self.recorder_switch.setAnimation(True)
        self.recorder_status_label = QLabel("Recorder is not running!")
        self.recorder_layout.addWidget(self.recorder_switch_icon)
        self.recorder_layout.addWidget(self.recorder_switch)
        self.recorder_layout.addWidget(self.recorder_status_label)
        self.recorder_layout.addItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))
        self.col2_layout.addLayout(self.recorder_layout)

        # Initially set the recording flag to inactive
        self.recording_active = False

        # Create the Ball Tracking Video Playback
        self.ball_tracking_media_player = QMediaPlayer()
        self.ball_tracking_video = QVideoWidget()
        # Add the ball tracking video to the second column Layout in first row
        self.col2_stacked_layout.addWidget(self.ball_tracking_video)
        # Set the ball tracking video Widget of column 2 to a minimum width of 28% of Screen Width
        self.ball_tracking_video.setFixedWidth(round(screen_size.width() * 0.28))
        # Set the content for the ball_tracking_media_player and reload after each play
        self.ball_tracking_media_player.setVideoOutput(self.ball_tracking_video)
        # Set the coordinates for the ball_tracking_image and ball_tracking_transparent_image once the media was loaded and the ball_tracking_video has assumed it's correct position
        self.ball_tracking_media_player.mediaStatusChanged.connect(self.onMediaStatusChanged)

        # Create a QLabel for the static image to keep showing after ball tracking video has played and hide it
        self.ball_tracking_image = QLabel()
        # Add the static image at the same position to the Tab layout as the ball tracking video
        self.col2_stacked_layout.addWidget(self.ball_tracking_image)
        # Initially hide the image to be able to see the video
        self.ball_tracking_image.hide()

        # Create a QLabel for the transparent overlay image
        self.ball_tracking_transparent_image = QLabel()
        # Add the static image at the same position to the Tab layout as the ball tracking video
        self.col2_stacked_layout.addWidget(self.ball_tracking_transparent_image)
        # Set the image to have a transparent background
        self.ball_tracking_transparent_image.setStyleSheet("background-color: transparent;")
        # Add the col2_stacked_layout to the col2_layout now that all elements were added to the stacked layout
        self.col2_layout.addLayout(self.col2_stacked_layout)

        # Initially hide the image
        self.ball_tracking_transparent_image.hide()
        # Load the Ball Tracking Video
        self.LoadBallTrackingContent()
        # Connect an action to show the image and re-load the video once it has finished playing
        self.ball_tracking_media_player.mediaStatusChanged.connect(
            lambda status: self.HandleMediaStatus(status, "ball_tracking", screen_size))

        # Create a Horizontal Box Layout for the Ball Tracking video controls
        self.ball_tracking_video_control_layout = QHBoxLayout()
        # Add the Layout of the Ball Tracking Video Controls to the column 2 layout in the second row
        self.col2_layout.addLayout(self.ball_tracking_video_control_layout)
        # Set the Alignment of this Layout to left
        self.ball_tracking_video_control_layout.setAlignment(Qt.AlignLeft)

        # Create the Play/Pause/Repeat button for the ball tracking video
        self.ball_tracking_play_button = QPushButton()
        # Add the Play Button to the ball tracking control layout (first column)
        self.ball_tracking_video_control_layout.addWidget(self.ball_tracking_play_button)
        # Set the button to 5% of the screen width
        self.ball_tracking_play_button.setFixedWidth(round(screen_size.width() * 0.05))
        # Set the Icon to the Play Icon
        self.ball_tracking_play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        # Link the button to the PlayContent function (play with a delay of 0)
        self.ball_tracking_play_button.clicked.connect(lambda: self.PlayContent("ball_tracking", 0))

        # Create label for the Speed Drop-down for the ball tracking video
        self.ball_tracking_speed_label = QLabel("Speed")
        # Add the Speed Label to the Ball Tracking control layout (second column)
        self.ball_tracking_video_control_layout.addWidget(self.ball_tracking_speed_label)
        # Set the Speed Label to 2.5% of the screen width
        self.ball_tracking_speed_label.setFixedWidth(round(screen_size.width() * 0.025))
        # Set the alignment of the Speed Label to Right
        self.ball_tracking_speed_label.setAlignment(Qt.AlignRight)

        # Create the Speed Dropdown-List for the ball tracking video
        self.ball_tracking_speed_combobox = QComboBox()
        # Add the Speed Dropdown-List to the ball tracking control layout (third column)
        self.ball_tracking_video_control_layout.addWidget(self.ball_tracking_speed_combobox)
        # Set the Speed Dropdown-List to 5% of the screen width
        self.ball_tracking_speed_combobox.setFixedWidth(round(screen_size.width() * 0.05))
        # Add the Speed options to the combobox and set the default value to 1
        self.ball_tracking_speed_combobox.addItems(["1/4", "1/2", "3/4", "1"])
        self.ball_tracking_speed_combobox.setCurrentText("1")
        # Connect the Dropdown-List to the setPlaybackSpeed function
        self.ball_tracking_speed_combobox.currentIndexChanged.connect(
            lambda index: self.SetPlaybackSpeed("ball_tracking", index))

        # Create label for the transparent overlay checkbox
        self.ball_tracking_transparent_checkbox_label = QLabel("Overlay")
        # Add the overlay checkbox label to the ball tracking control layout (fourth column)
        self.ball_tracking_video_control_layout.addWidget(self.ball_tracking_transparent_checkbox_label)
        # Set the alignment of the Overlay Label to Right
        self.ball_tracking_transparent_checkbox_label.setAlignment(Qt.AlignRight)

        # Create a checkbox to toggle the transparent image
        self.ball_tracking_transparent_checkbox = QCheckBox()
        # Add the checkbox to the ball tracking control layout (fifth column)
        self.ball_tracking_video_control_layout.addWidget(self.ball_tracking_transparent_checkbox)
        # Toggle the transparent overlay when the checkbox is checked
        self.ball_tracking_transparent_checkbox.stateChanged.connect(
            lambda index: self.HandleTransparentMediaStatus(screen_size))

        # Create the Ball Tracking Video Save button
        self.ball_tracking_save_button = QPushButton("Save Video")
        # Add the button to the ball tracking control layout
        self.ball_tracking_video_control_layout.addWidget(self.ball_tracking_save_button)
        # Connect the SaveContent function to the button
        self.ball_tracking_save_button.clicked.connect(lambda index: self.SaveContent("Ball_Tracking", self.lane_number))

        # Create the pins Video Playback
        self.pins_media_player = QMediaPlayer()
        self.pins_video = QVideoWidget()
        # Add the Pins Video to the tab layout (first row, third column, spanning three rows)
        self.col3_layout.addWidget(self.pins_video)
        # Fix the Width of the pins video to 35% of the screen width
        self.pins_video.setFixedWidth(round(screen_size.width() * 0.35))
        # Set the video output for the pins video and reload it after each play
        self.pins_media_player.setVideoOutput(self.pins_video)
        # Load the video content
        self.LoadPinsContent()

        # Create a QLabel for the static image to keep showing after pins video has played
        self.pins_image = QLabel()
        # Add the static pins image to the tab layout at the same position as the pins video
        self.col3_layout.addWidget(self.pins_image)
        # Initially hide the image
        self.pins_image.hide()

        # Create a GridLayout for the pins video control buttons
        self.pins_video_control_layout = QHBoxLayout()
        # Add the Layout of the Pins Video Controls to the column 3 layout (second row)
        self.col3_layout.addLayout(self.pins_video_control_layout)
        # Set the Alignment of this Layout to left
        self.pins_video_control_layout.setAlignment(Qt.AlignLeft)

        # Create the Play/Pause/Repeat button for the ball tracking video
        self.pins_play_button = QPushButton()
        # Add the Pins Play button to the pins video control layout (first column)
        self.pins_video_control_layout.addWidget(self.pins_play_button)
        # Set the button to 5% of the screen width
        self.pins_play_button.setFixedWidth(round(screen_size.width() * 0.05))
        # Set the icon to the Play Icon
        self.pins_play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        # Link the button to the PlayContent function (play with a delay of 0)
        self.pins_play_button.clicked.connect(lambda: self.PlayContent("pins", 0))

        # Connect an action to show the image and re-load the video once it has finished playing
        self.pins_media_player.mediaStatusChanged.connect(
            lambda status: self.HandleMediaStatus(status, "pins", screen_size))

        # Create label for the Speed Drop-down for the pins video
        self.pins_speed_label = QLabel("Speed")
        # Add the speed label to the pins video control layout (second column)
        self.pins_video_control_layout.addWidget(self.pins_speed_label)
        # Set the Label to align right
        self.pins_speed_label.setAlignment(Qt.AlignRight)
        # Set the label to 2.5% of the screen width
        self.pins_speed_label.setFixedWidth(round(screen_size.width() * 0.025))

        # Create the Speed Dropdown-List for the pins video
        self.pins_speed_combobox = QComboBox()
        # Add the speed combobox to the pins video control layout (third column)
        self.pins_video_control_layout.addWidget(self.pins_speed_combobox)
        # Set the label to 3% of the screen width
        self.pins_speed_combobox.setFixedWidth(round(screen_size.width() * 0.03))
        # Set the speed options
        self.pins_speed_combobox.addItems(["1/4", "1/2", "3/4", "1"])
        # Set the default value of the pins speed combobox to "1/2"
        self.pins_speed_combobox.setCurrentText("1/2")
        # Run the SetPlaybackSpeed function, to default the speed of the pin video to 0.5
        self.SetPlaybackSpeed("pins", self.pins_speed_combobox.currentIndex())
        # Connect the dropdown-list to the SetPlaybackSpeed function
        self.pins_speed_combobox.currentIndexChanged.connect(
            lambda index: self.SetPlaybackSpeed("pins", index))

        # Create the Pins Video Save button
        self.pins_save_button = QPushButton("Save Video")
        # Add the button to the ball tracking control layout
        self.pins_video_control_layout.addWidget(self.pins_save_button)
        # Connect the SaveContent function to the button
        self.pins_save_button.clicked.connect(lambda index: self.SaveContent("Pins", self.lane_number))

        # Insert a Spacer between the pin controls and the GUI table for 5% of screen height
        self.spacer_item_col3_control = QSpacerItem(20, round(screen_size.height() * 0.05))
        self.col3_layout.addItem(self.spacer_item_col3_control)

        # Generate the tab widget to display different players in different tabs
        self.table_tab_widget = QTabWidget()
        # Add this tab widget to the third column
        self.col3_layout.addWidget(self.table_tab_widget)
        # Initially hide the tab widget
        self.table_tab_widget.setVisible(False)

        # Set a placeholder for the tab/table when it is hidden
        self.overview_table_placeholder = QLabel()
        # Set the height to the 40 % of the screen (which will also be the size of the table
        self.overview_table_placeholder.setFixedHeight(round(screen_size.height() * 0.4))
        # Add the placeholder to the column 3 layout
        self.col3_layout.addWidget(self.overview_table_placeholder)

        # Update field visibility (to hide the record_mode_inputs) upon boot of the program
        self.updateFieldVisibility()

        # Define the Montior to be the MonitorWorker class
        self.tracking_data_updater = TrackingDataUpdater(self, self.lane_number)

        # Connect the Ball Tracker signals
        signal_router.tracking_data_available.connect(self.tracking_data_updater.ProcessTrackingData)
        signal_router.tracking_unsuccessful.connect(self.tracking_data_updater.ProcessTrackingData)

        # Link the GUI Table update signal also to the MonitorWorker and send it to the UpdateTable function
        self.tracking_data_updater.update_table_signal.connect(self.UpdateTable)
        # Link the GUI Table reset signal also to the MonitorWorker and send it to the ResetTable function
        self.tracking_data_updater.reset_table_signal.connect(self.ResetTable)
        # Link the GUI Table calculate mean and sd signal to the MonitorWorker and send it to the CalcMeanAndSD function
        self.tracking_data_updater.calculate_mean_and_sd_signal.connect(self.CalcMeanAndSD)


        # Connect the finish tracking signal also to reloading and playing the video contents
        signal_router.tracking_data_available.connect(self.LoadBallTrackingContent)
        signal_router.tracking_data_available.connect(self.LoadPinsContent)
        # Play the contents with some time delay
        signal_router.tracking_data_available.connect(lambda: self.PlayContent("ball_tracking", 50))
        signal_router.tracking_data_available.connect(lambda: self.PlayContent("pins", 2000))

        # Connect the signal to load content when tracking was unsuccessful
        signal_router.tracking_unsuccessful.connect(self.LoadBallTrackingUnsuccessfulContent)
        signal_router.tracking_unsuccessful.connect(self.LoadPinsContent)
        # Play the contents with some time delay
        signal_router.tracking_unsuccessful.connect(lambda: self.PlayContent("pins", 2000))


    # Function to update the visibility of the player fields and the table as well as the overlay checkbox
    def updateFieldVisibility(self):
        # When the Record Mode Button is checked, display the player controls and the overlay checkbox
        if self.record_mode_button.isChecked():
            self.record_mode_placeholder.setVisible(False)
            self.record_mode_inputs.setVisible(True)
            self.ball_tracking_transparent_checkbox.setVisible(True)
            self.ball_tracking_transparent_checkbox_label.setVisible(True)

        # When the Single Shot Button is checked, hide the player controls and the overlay checkbox and clear the Tab Widget for the GUI table
        elif self.single_shot_button.isChecked():
            self.record_mode_placeholder.setVisible(True)
            self.record_mode_inputs.setVisible(False)
            self.ball_tracking_transparent_checkbox.setVisible(False)
            self.ball_tracking_transparent_checkbox_label.setVisible(False)
            self.table_tab_widget.clear()
            self.table_tab_widget.setVisible(False)
            self.overview_table_placeholder.setVisible(True)
            # Delete any selected_player from the record mode
            self.selected_player = None

    # Function to play the video contents
    def PlayContent(self, object, delay):
        # If the Balltracking Video is playing, pause it and update the play/pause button to show the play sign
        if object == "ball_tracking" and self.ball_tracking_media_player.state() == QMediaPlayer.PlayingState:
            self.ball_tracking_media_player.pause()
            self.ball_tracking_play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        # If the Balltracking Video is not playing and the image is visible, hide the image, show the video and play it after the delay. Update the play/pause button to show the pause sign
        elif object == "ball_tracking":
            if self.ball_tracking_image.isVisible():
                self.ball_tracking_video.show()
                self.ball_tracking_image.hide()
            QTimer.singleShot(delay, self.ball_tracking_media_player.play)
            self.ball_tracking_play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

        # If the Pin Video is playing, pause it and update the play/pause button to show the play sign
        elif object == "pins" and self.pins_media_player.state() == QMediaPlayer.PlayingState:
            self.pins_media_player.pause()
            self.pins_play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

        # If the Pin Video is not playing and the image is visible, hide the image, show the video and play it after the delay. Update the play/pause button to show the pause sign
        elif object == "pins":
            if self.pins_image.isVisible():
                self.pins_video.show()
                self.pins_image.hide()
            QTimer.singleShot(delay, self.pins_media_player.play)
            self.pins_play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    # Function to show the static image and hide the video once finished playing and reload the video file
    def HandleMediaStatus(self, status, object, screen_size):
        # If the balltracking video has finished playing, show the repeat sign on the button, hide the video and show the image as well as reload the video
        if status == QMediaPlayer.EndOfMedia and object == "ball_tracking":
            self.ball_tracking_play_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
            self.ball_tracking_video.hide()
            self.ball_tracking_image.show()
            pixmap = QPixmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'tracked_new_' + str(self.lane_number) + '.png'))
            scaled_pixmap = pixmap.scaled(self.ball_tracking_video.geometry().width(),
                                          self.ball_tracking_video.geometry().height(), Qt.KeepAspectRatio)
            self.ball_tracking_image.setPixmap(scaled_pixmap)
            self.ball_tracking_image.setAlignment(Qt.AlignCenter)
            self.LoadBallTrackingContent()

        # If the pin video has finished playing, show the repeat sign on the button, hide the video and show the image as well as reload the video
        if status == QMediaPlayer.EndOfMedia and object == "pins":
            self.pins_play_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
            self.pins_video.hide()
            self.pins_image.show()
            pixmap = QPixmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'pins_new_' + str(self.lane_number) + '.png'))
            scaled_pixmap = pixmap.scaledToWidth(round(screen_size.width() * 0.35))
            self.pins_image.setPixmap(scaled_pixmap)
            self.LoadPinsContent()

    # Function to toggle the transparent overlay image
    def HandleTransparentMediaStatus(self, screen_size):
        # If the overlay button is checked, display the transparent overlay image
        if self.ball_tracking_transparent_checkbox.isChecked():
            self.ball_tracking_transparent_image.show()
            pixmap = QPixmap(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos',
                                          'tracked_new_transparent_' + str(self.lane_number) + '_inuse.png'))
            scaled_pixmap = pixmap.scaledToWidth(round(screen_size.width() * 0.28))
            self.ball_tracking_transparent_image.setPixmap(scaled_pixmap)
        # If the overlay button is un-checked, hide the transparent overlay image
        elif not self.ball_tracking_transparent_checkbox.isChecked():
            self.ball_tracking_transparent_image.hide()

    # Function to load the video from the ball tracking camera
    def LoadBallTrackingContent(self):
        # Load the media content for the ball tracking video
        ball_tracking_video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos',
                                                'tracked_new_' + str(self.lane_number) + '.mp4')
        ball_tracking_content = QMediaContent(QUrl.fromLocalFile(ball_tracking_video_path))
        self.ball_tracking_media_player.setMedia(ball_tracking_content)

    # Function to load the video from the pins camera
    def LoadPinsContent(self):
        # Load the media content for the pin video
        pins_video_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'pins_new_' + str(self.lane_number) + '.mp4')
        pins_content = QMediaContent(QUrl.fromLocalFile(pins_video_path))
        self.pins_media_player.setMedia(pins_content)

    # Function to load the animation if tracking was unsuccessful
    def LoadBallTrackingUnsuccessfulContent(self):
        # Path to the .gif file
        gif_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'templates', 'tracking_unsuccessful.gif')

        # Set the Movie
        gif_movie = QMovie(gif_path)
        # Set the QMovie object to the QLabel
        self.ball_tracking_image.setMovie(gif_movie)

        # Optionally, you can set the alignment for better display
        self.ball_tracking_image.setAlignment(Qt.AlignCenter)

        # Start the gif animation
        gif_movie.start()

        # Make sure the QLabel is visible
        self.ball_tracking_image.show()
        self.ball_tracking_video.hide()

    # Function to save any video file
    def SaveContent(self, content, lane_number):
        # Define the variables of setting a name, game number and frame number
        name = None
        game = None
        frame = None

        # If there is a player active and selected, obtain its name, current game and current frame
        if self.selected_player:
            self.GetSelectedPlayerGameAndFrame()
            name = self.selected_player
            game = self.selected_player_current_game
            frame = self.selected_player_current_frame

        # Call and show the Video saving window with the optionally obtained name, game and frame variable
        self.video_saving_window = VideoSavingWindow(content, lane_number, name, game, frame)
        self.video_saving_window.show()

    # Function to alter the playback speed of the videos
    def SetPlaybackSpeed(self, object, index):
        # for the ball tracking video, get the selection from the speed combobox
        if object == "ball_tracking":
            speed_text = self.ball_tracking_speed_combobox.currentText()
            # Map the speed text to a numeric value
            speed_mapping = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75, "1": 1.0}
            # Get the speed and set the default to 1
            self.playback_speed = speed_mapping.get(speed_text, 1.0)
            # Apply the playback speed to the media player
            self.ball_tracking_media_player.setPlaybackRate(self.playback_speed)

        if object == "pins":
            # for the pin video, get the selection from the speed combobox
            speed_text = self.pins_speed_combobox.currentText()
            # Map the speed text to a numeric value
            speed_mapping = {"1/4": 0.25, "1/2": 0.5, "3/4": 0.75, "1": 1.0}
            # Get the speed and set the default to 0.5
            self.pins_playback_speed = speed_mapping.get(speed_text, 1.0)
            # Apply the playback speed to the media player
            self.pins_media_player.setPlaybackRate(self.pins_playback_speed)

    # Function to create and update the player fields in Record mode
    def updatePlayerFields(self):
        # Obtain the number of players and define the frame names
        player_count = self.player_count_spinbox.value()
        frame_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10.2, 10.3]

        # Clear existing player fields excluding the first row as these are the headers
        for i in reversed(range(self.player_layout.count())):
            self.player_layout.itemAt(i).widget().setParent(None)

        # Create the labels for "Player", "Game" and "Frame" and add it to the player_layout (first row, first, second and third column respectively)
        player_label = QLabel("Player")
        self.player_layout.addWidget(player_label, 0, 0)
        game_label = QLabel("Game")
        self.player_layout.addWidget(game_label, 0, 1)
        frame_label = QLabel("Frame")
        self.player_layout.addWidget(frame_label, 0, 2)

        # Generate the Player name, Game and Frame fields for each player
        for i in range(player_count):
            player_name = QLineEdit()
            player_game = QSpinBox()
            player_game.setMinimum(1)
            player_frame = QComboBox()
            for frame in frame_values:
                player_frame.addItem(str(frame))
            self.player_layout.addWidget(player_name, i + 1, 0)
            self.player_layout.addWidget(player_game, i + 1, 1)
            self.player_layout.addWidget(player_frame, i + 1, 2)

    # Function to define the ball_tracking image and it's transparent overlay to the same size as the video
    def onMediaStatusChanged(self, status):
        # Once the ball tracking video is loaded, set the image and transparent image to the same geometry
        if status == QMediaPlayer.LoadedMedia:
            self.ball_tracking_image.setGeometry(self.ball_tracking_video.geometry())
            self.ball_tracking_transparent_image.setGeometry(self.ball_tracking_video.geometry())

    # Function to select the next player
    def nextPlayer(self):
        # Get all player names from player_name fields
        player_names = []
        for i in range(self.player_layout.rowCount()):
            item = self.player_layout.itemAtPosition(i, 0)
            if item and isinstance(item.widget(), QLineEdit):
                player_names.append(item.widget().text())

        # Select the next player or the first player if none is selected yet
        if self.selected_player is None or self.selected_player not in player_names:
            self.selected_player = player_names[0]
        # Select the first player if the last player is selected and increase the frame combobox to the next frame
        elif self.selected_player == player_names[-1]:
            self.selected_player = player_names[0]
            for i in range(self.player_layout.rowCount()):
                item = self.player_layout.itemAtPosition(i, 2)
                if item and isinstance(item.widget(), QComboBox):
                    combobox = item.widget()
                    current_index = combobox.currentIndex()
                    next_index = (current_index + 1) % combobox.count()
                    combobox.setCurrentIndex(next_index)

        # Otherwise just select the next player
        else:
            current_index = player_names.index(self.selected_player)
            self.selected_player = player_names[(current_index + 1) % len(player_names)]

        # Highlight the corresponding player_name field in green
        for i in range(self.player_count_spinbox.value() + 1):  # Adding one for the header row
            item = self.player_layout.itemAtPosition(i, 0)
            player_name_field = item.widget()
            if player_name_field.text() == self.selected_player:
                player_name_field.setStyleSheet("background-color: darkgreen;")
            else:
                player_name_field.setStyleSheet("")
        # Obtain the tab of the selected_player
        for i in range(self.table_tab_widget.count()):
            if self.table_tab_widget.tabText(i) == self.selected_player:
                self.table_tab_widget.setCurrentIndex(i)
                break
        # Obtain the current frame and game of the selected player (to be used for storring tracking data)
        self.GetSelectedPlayerGameAndFrame()

    # Function to set the next game for all players
    def nextGame(self):
        # Obtain all player names before starting a new game
        player_names = []
        for i in range(self.player_count_spinbox.value() + 1):
            item = self.player_layout.itemAtPosition(i, 0)
            if item and isinstance(item.widget(), QLineEdit):
                player_names.append(item.widget().text())
        # Set the selected_player to the first one for the new game
        self.selected_player = player_names[0]
        # Highlight the corresponding player_name field in green
        for i in range(self.player_count_spinbox.value() + 1):
            item = self.player_layout.itemAtPosition(i, 0)
            player_name_field = item.widget()
            if player_name_field.text() == self.selected_player:
                player_name_field.setStyleSheet("background-color: darkgreen;")
            else:
                player_name_field.setStyleSheet("")
        # Increase all Game Spinboxes by one, reset the frame combobox to frame 1
        for i in range(self.player_count_spinbox.value() + 1):
            game_spinbox = self.player_layout.itemAtPosition(i, 1)
            if game_spinbox and isinstance(game_spinbox.widget(), QSpinBox):
                game_spinbox.widget().setValue(game_spinbox.widget().value() + 1)
            frame_combobox = self.player_layout.itemAtPosition(i, 2)
            if frame_combobox and isinstance(frame_combobox.widget(), QComboBox):
                combobox = frame_combobox.widget()
                combobox.setCurrentIndex(0)

        # Obtain the tab of the selected_player
        for i in range(self.table_tab_widget.count()):
            if self.table_tab_widget.tabText(i) == self.selected_player:
                self.table_tab_widget.setCurrentIndex(i)
                break

        # Obtain the current frame and game of the selected player (to be used for storring tracking data)
        self.GetSelectedPlayerGameAndFrame()

    # Function to enable/disable editing of the player fields once they were confirmed and change the button text
    def confirmPlayers(self):
        # If the button shows "Confirm Players", set it to "Edit Players", show the next player and new game button and set the input fiels to read only.
        if self.edit_player_button.text() == "Confirm Players":
            self.edit_player_button.setText("Edit Players")
            self.next_player_button.setVisible(True)
            self.new_game_button.setVisible(True)
            for i in range(self.player_layout.rowCount()):
                item = self.player_layout.itemAtPosition(i, 0)
                if item and isinstance(item.widget(), QLineEdit):
                    item.widget().setReadOnly(True)
            self.player_count_spinbox.setReadOnly(True)
            self.overview_table_placeholder.setVisible(False)
            self.GenerateOverviewTables()
        # If the button shows "Edit Players", set it to "Confirm Players", hide the next player and new game button and set the input fiels to editable.
        else:
            self.edit_player_button.setText("Confirm Players")
            self.next_player_button.setVisible(False)
            self.new_game_button.setVisible(False)
            for i in range(self.player_layout.rowCount()):
                item = self.player_layout.itemAtPosition(i, 0)
                if item and isinstance(item.widget(), QLineEdit):
                    item.widget().setReadOnly(False)
            self.player_count_spinbox.setReadOnly(False)
            # remove the selected_player and set all to not green
            self.selected_player = None
            for i in range(self.player_count_spinbox.value() + 1):
                item = self.player_layout.itemAtPosition(i, 0)
                player_name_field = item.widget()
                player_name_field.setStyleSheet("")

    # Funciton to generate the Overview Table
    def GenerateOverviewTables(self):
        # Show the tab widget
        self.table_tab_widget.setVisible(True)

        # Clear existing tabs
        self.table_tab_widget.clear()

        # Get player names
        player_names = []
        for i in range(self.player_count_spinbox.value() + 1):
            item = self.player_layout.itemAtPosition(i, 0)
            if item and isinstance(item.widget(), QLineEdit):
                player_names.append(item.widget().text())

        # Define the headers and frame numbers
        headers = ["Frame", "Foul Line", "Arrows", "Breakpoint", "Entry Point", "Speed"]
        frame_numbers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10.2", "10.3", "Average", "Std. Dev."]

        # For every player, generate a table, set the headers and frames
        for player_name in player_names:
            # Create a new table for each player
            table = QTableWidget()
            table.setColumnCount(len(headers))
            table.setRowCount(len(frame_numbers))
            table.setHorizontalHeaderLabels(headers)
            for row in range(len(frame_numbers)):
                item = QTableWidgetItem(frame_numbers[row])
                table.setItem(row, 0, item)
                if row >= table.rowCount() - 2:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    font = QFont()
                    font.setBold(True)
                    item.setFont(font)
            table.setFixedHeight(table.verticalHeader().defaultSectionSize() * (len(frame_numbers) + 1))

            # Add the table to the tab widget with the player's name as the tab label
            self.table_tab_widget.addTab(table, player_name)

        # Hide the placeholder
        self.overview_table_placeholder.setVisible(False)

    # Function to update data in the table
    def UpdateTable(self, data_update):
        # This function is executed upon receiving the update_table signal from the MonitorThread(). First obtain the player name, their game number, the frame index and the data
        current_player_name = data_update['player_name']
        current_game = data_update['player_game']
        frame_index = data_update['frame_index']
        data = data_update['data']

        # Define variables to hold all player names and games from the GUI and store the values into that variable
        current_players_and_games = []
        player_item = None
        game_item = None
        for i in range(self.player_count_spinbox.value() + 1):
            if isinstance(self.player_layout.itemAtPosition(i, 0).widget(), QLineEdit):
                player_item = self.player_layout.itemAtPosition(i, 0).widget().text()
            if isinstance(self.player_layout.itemAtPosition(i, 1).widget(), QSpinBox):
                game_item = self.player_layout.itemAtPosition(i, 1).widget().value()
            if player_item and game_item:
                current_players_and_games.append((player_item, game_item))
        # If the incoming player name and game matches an available player name and the game is current, select the players table and input the data
        if (current_player_name, current_game) in current_players_and_games:
            # Find the tab index for the player
            for i in range(self.table_tab_widget.count()):
                if self.table_tab_widget.tabText(i) == current_player_name:
                    table_widget = self.table_tab_widget.widget(i)
                    break
            # Populate the table with the new data
            table_widget.setItem(frame_index, 1, QTableWidgetItem(str(data['position_at_foul_line'])))
            table_widget.setItem(frame_index, 2, QTableWidgetItem(str(data['position_at_arrows'])))
            table_widget.setItem(frame_index, 3, QTableWidgetItem(str(data['position_at_breakpoint'])))
            table_widget.setItem(frame_index, 5, QTableWidgetItem(str(data['ball_speed'])))

    # Function to reset the table before a new data load
    def ResetTable(self):
        # Clear the table data except the headers and the first column
        for i in range(self.table_tab_widget.count()):
            table_widget = self.table_tab_widget.widget(i)
            for row in range(table_widget.rowCount()):
                for col in range(1, table_widget.columnCount()):
                    table_widget.setItem(row, col, QTableWidgetItem(""))

    # Function to calculate the mean and the standard deviation
    def CalcMeanAndSD(self):
        try:
            # for each sheet in the table, obtain the amount of rows and columns
            for i in range(self.table_tab_widget.count()):
                table_widget = self.table_tab_widget.widget(i)
                row_count = table_widget.rowCount()
                column_count = table_widget.columnCount()
                # For each column in the current sheet of the table, extract the values for each row, expect the last two rows
                for col in range(1, column_count):
                    values = []
                    for row in range(row_count - 2):  # Exclude the last two rows for calculations as they show the calculated values
                        item = table_widget.item(row, col)
                        # Only use the value of a cell if it is not "n/a" or "Gutter"
                        if item is not None and item.text() != "n/a" and item.text() != "Gutter" and item.text().strip():
                            try:
                                values.append(float(item.text()))
                            except ValueError:
                                pass
                    # Calculate SD and mean if there are some values present
                    if len(values) > 0:
                        mean_value = sum(values) / len(values)
                        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
                        sd_value = (variance ** 0.5)
                        table_widget.setItem(row_count - 2, col, QTableWidgetItem(f"{mean_value:.1f}"))
                        table_widget.setItem(row_count - 1, col, QTableWidgetItem(f"{sd_value:.1f}"))
        except Exception as e:
            # Log the error or handle it appropriately
            QMessageBox.critical(None, "Mean or Standard Deviation not calculated", "There was an error in calculating the Mean or Standard Deviation. ", e)

    # Function to obtain the current game and frame for a selected player
    def GetSelectedPlayerGameAndFrame(self):
        # Define the frame values
        frame_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10.2, 10.3]
        # Obtain selected_player game and frame
        for i in range(self.player_count_spinbox.value() + 1):
            item = self.player_layout.itemAtPosition(i, 0)
            player_name_field = item.widget()
            if player_name_field.text() == self.selected_player:
                self.selected_player_current_game = self.player_layout.itemAtPosition(i, 1).widget().value()
                self.selected_player_current_frame = frame_values[
                    self.player_layout.itemAtPosition(i, 2).widget().currentIndex()]

    # Function to toggle the ball recorder on or off
    def ToggleRecorder(self, checked):
        # If the switch is turned on, start the recorder
        if checked:
            self.StartRecorder()
        # If the switch is turned off, stop the recorder
        else:
            self.StopRecorder()

    # Function to update the Recording label to let the user know what the recorder is doing at the moment
    @pyqtSlot(str)
    def UpdateRecorderStatus(self, status):
        if status == "recording":
            self.recorder_status_label.setText("Recording...")
            self.recorder_status_label.setStyleSheet("color: red;")

        if status == "generating video":
            self.recorder_status_label.setText("Saving Videos...")
            self.recorder_status_label.setStyleSheet("color: orange;")

        if status == "tracking":
            self.recorder_status_label.setText("Tracking Shot...")
            self.recorder_status_label.setStyleSheet("color: blue;")

        if status == "resetting":
            self.recorder_status_label.setText("Resetting Recorder...")
            self.recorder_status_label.setStyleSheet("color: purple;")

        if status == "idle":
            self.recorder_status_label.setText("Recorder idle. Ready for next shot!")
            self.recorder_status_label.setStyleSheet("color: #70ff8f;")

        if status == "recorder offline":
            self.recorder_status_label.setText("Recorder is not running!")
            self.recorder_status_label.setStyleSheet("")

    # Function to start the recorder
    def StartRecorder(self):
        # Load the settings for the current lane
        config = configparser.ConfigParser()
        if os.path.exists(f'settings_lane_{self.lane_number}.cfg'):
            config.read(f'settings_lane_{self.lane_number}.cfg')
        else:
            QMessageBox.critical(None, "No Settings found", "No settings for lane " + str(self.lane_number) + " were found. Please go to settings and choose the lane to autogenerate the settings file. Afterwards reboot the programm.")
            return

        # Start the recorder thread if the recording is not already running and set the record mode to active
        if not self.recording_active:
            # Generate the Recorder Thread and start the Thread
            self.recorder_worker = RecorderWorker(self.lane_number)
            self.recorder_worker.start()
            # Connect the recorder_status signal to the UpdateRecorderStatus slot
            self.recorder_worker.recorder_status.connect(self.UpdateRecorderStatus)
            # Set the recording mode to active
            self.recording_active = True

    # Function to stop the recorder
    def StopRecorder(self):
        # If the record mode is active and the thread exists, stop it and set the mode to inactive
        if self.recording_active and self.recorder_worker:
            self.recorder_worker.StopMonitoring()
            self.recorder_worker.wait()
            self.recorder_worker = None
            self.recording_active = False

    # Function to Disconnect the Signals and turn off the recorder when the program is closed
    def DisconnectSignalsAndStopRecorder(self):
        # Disconnect signals to avoid potential issues
        self.tracking_data_updater.update_table_signal.disconnect()
        self.tracking_data_updater.reset_table_signal.disconnect()
        self.tracking_data_updater.calculate_mean_and_sd_signal.disconnect()
        signal_router.tracking_data_available.disconnect()
        signal_router.tracking_unsuccessful.disconnect()

        # Stop the Recorder if it is running
        self.StopRecorder()

    # Function to run when the program is closed
    def closeEvent(self, event):
        self.DisconnectSignalsAndStopRecorder()
        self.StopRecorder()
        event.accept()

# Define a class for the pop up window to input values to save some videos
class VideoSavingWindow(QWidget):
    def __init__(self, content, lane_number, name=None, game=None, frame=None):
        super().__init__()

        # Obtain the variables for video saving
        self.content = content
        self.lane_number = lane_number
        self.name = name
        self.game = str(game)
        self.frame = str(frame)

        # Set window title
        self.setWindowTitle('Save Video')

        # Create layout
        layout = QVBoxLayout()

        # Create and add labels and line edits for name, game, frame, and comment
        self.name_label = QLabel('Name:')
        self.name_edit = QLineEdit(self.name if self.name else '')
        self.game_label = QLabel('Game:')
        self.game_edit = QLineEdit(self.game if self.game else '')
        self.frame_label = QLabel('Frame:')
        self.frame_edit = QLineEdit(self.frame if self.frame else '')
        self.comment_label = QLabel('Comment:')
        self.comment_edit = QLineEdit()

        layout.addWidget(self.name_label)
        layout.addWidget(self.name_edit)
        layout.addWidget(self.game_label)
        layout.addWidget(self.game_edit)
        layout.addWidget(self.frame_label)
        layout.addWidget(self.frame_edit)
        layout.addWidget(self.comment_label)
        layout.addWidget(self.comment_edit)

        # Create and add a button to confirm
        self.save_button = QPushButton('Save')
        self.save_button.clicked.connect(self.SaveFile)
        layout.addWidget(self.save_button)

        # Set the layout to the window
        self.setLayout(layout)

    # Define a function to save the files
    def SaveFile(self):
        # Get input values
        name = self.name_edit.text().strip()
        game = self.game_edit.text().strip()
        frame = self.frame_edit.text().strip()
        comment = self.comment_edit.text().strip()

        # Check if mandatory fields are filled
        if not name or not game or not frame:
            QMessageBox.critical(self, 'Error', 'Name, Game, and Frame fields are mandatory!')
            return

        # Obtain today's date for saving
        today_date = datetime.datetime.now().strftime('%d-%b-%Y')

        # Save the video with or without the comment, depending if a comment exists
        if not comment:
            # Define the destination file name
            destination_file = f"{today_date}_{name}_Game_{game}_Frame_{frame}_{self.content}.mp4"
        else:
            destination_file = f"{today_date}_{name}_Game_{game}_Frame_{frame}_{self.content}_{comment}.mp4"

        # Full path to the destination file
        save_directory = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Saved_Videos')
        full_destination_path = os.path.join(save_directory, destination_file)

        # Define the source file (Ball Tracking vs. Pins camera)
        if self.content == "Ball_Tracking":
            source_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'tracked_new_' + str(self.lane_number) + '.mp4')

        elif self.content == "Pins":
            source_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos', 'pins_new_' + str(self.lane_number) + '.mp4')

        try:
            # Copy the file
            shutil.copy2(source_file, full_destination_path)
            QMessageBox.information(self, 'Success', f'Video saved!')
            self.close()
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Video saving unsuccessful: {str(e)}')
