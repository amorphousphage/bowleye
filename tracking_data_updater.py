# import used packages
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
import json
import time
import datetime
import os

# Create a class to run the Tracking Data updater in
class TrackingDataUpdater(QObject):
    # Define the signal to be sent when a data update is available
    update_table_signal = pyqtSignal(dict)  # Define the signal
    # Define the signal to be sent when a data update is available and table needs to be reset first
    reset_table_signal = pyqtSignal()
    # Define the signal to be sent when the last row of complete_tracking_data was sent to initiate calculation of average and relative standard deviation
    calculate_mean_and_sd_signal = pyqtSignal()

    def __init__(self, main_window, lane_number):
        super().__init__()
        self.main_window = main_window
        self.lane_number = lane_number

    @pyqtSlot()
    def ProcessTrackingData(self):
        # Define today's date
        today_date = datetime.datetime.now().strftime("%d-%b-%Y")
        # Define the file name for today's complete_tracking_data.json file
        complete_tracking_data_filename = f'complete_tracking_data_lane_{self.lane_number}_{today_date}.json'

        # Load the tracking data file
        with open('ball_tracking_data_lane_' + str(self.lane_number) + '.json', 'r') as f:
            ball_tracking_data = json.load(f)

        # If we have an active player, append the data with the player name, current frame and current game
        if self.main_window.selected_player:
            self.main_window.GetSelectedPlayerGameAndFrame()

            # Allocate data to the active player
            player_name = self.main_window.selected_player
            current_frame = self.main_window.selected_player_current_frame
            current_game = self.main_window.selected_player_current_game
            # Save to a separate JSON line
            new_tracked_data = {
                'player_name': player_name,
                'current_frame': current_frame,
                'current_game': current_game,
                'ball_speed': ball_tracking_data['ball_speed'],
                'position_at_foul_line': ball_tracking_data['position_at_foul_line'],
                'position_at_arrows': ball_tracking_data['position_at_arrows'],
                'position_at_breakpoint': ball_tracking_data['position_at_breakpoint']
            }
            # Read the existing complete tracking data
            if os.path.exists(complete_tracking_data_filename):
                with open(complete_tracking_data_filename, 'r') as f:
                    complete_tracking_data = [json.loads(line) for line in f]
            else:
                complete_tracking_data = []

            if not any(entry['player_name'] == player_name and entry['current_game'] == current_game and entry['current_frame'] == current_frame for entry in complete_tracking_data):
                with open(complete_tracking_data_filename, 'a') as f:
                    json.dump(new_tracked_data, f)
                    f.write('\n')

            # Reset the GUI table to reload the data
            self.reset_table_signal.emit()
            # Reload the stored complete tracking data (incl. names, games, frames etc)
            if os.path.exists(complete_tracking_data_filename):
                with open(complete_tracking_data_filename, 'r') as f:
                    complete_tracking_data = [json.loads(line) for line in f]
            else:
                complete_tracking_data = []
            # For every entry in that file, obtain the data
            for entry in complete_tracking_data:
                player_name = entry['player_name']
                current_game = entry['current_game']
                frame_index = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "10.2", "10.3"].index(
                    str(entry['current_frame']))

                data_update = {
                    'player_name': player_name,
                    'player_game': current_game,
                    'frame_index': frame_index,
                    'data': {
                        'position_at_foul_line': entry['position_at_foul_line'],
                        'position_at_arrows': entry['position_at_arrows'],
                        'position_at_breakpoint': entry['position_at_breakpoint'],
                        'ball_speed': entry['ball_speed']
                    }
                }
                # Send the data from the row to the UpdateTable function of the GUI using the update_table signal
                self.update_table_signal.emit(data_update)

            # Emit the signal to calculate mean and SD after processing all entries
            self.calculate_mean_and_sd_signal.emit()
