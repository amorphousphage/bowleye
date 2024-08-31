# import used packages
import sys
import traceback
from PyQt5.QtWidgets import QMainWindow, QTabWidget, QDesktopWidget, QApplication, QMessageBox, QPushButton, QWidget, QVBoxLayout, QDialog, QLabel
from PyQt5.QtCore import pyqtSignal, QRect, Qt, QThread, QSize
from PyQt5.QtGui import QMovie
from lane_tab import LaneTab
from settings_tab import SettingsTab


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        # Define the window title
        self.setWindowTitle("BowlEye - Bowling Ball Tracking Software")

        # Define a main Widget and Layout to host the sub-layouts
        main_widget = QWidget()
        main_layout = QVBoxLayout()

        # Create the tab widget
        self.tabWidget = QTabWidget()
        main_layout.addWidget(self.tabWidget)

        # Create settings tabs (includes loading the settings saved)
        self.settings_tab = SettingsTab()
        self.tabWidget.addTab(self.settings_tab, "Settings")

        # Load the lane numbers from the lanes.cfg configuration file (existance of this file is checked within SettingsTab())
        with open('lanes.cfg', 'r') as file:
            lane_numbers = file.read().splitlines()

        # Create the lane tabs
        self.lane_tabs = []
        for lane in lane_numbers:
            self.lane_tab = LaneTab(lane)
            self.lane_tabs.append(self.lane_tab)
            self.tabWidget.addTab(self.lane_tab, "Lane " + str(lane))

        # Move the Settings tab from first to last position (it needed to be created first in order to load the correct lanes for the lane tab, hence its subsequent moving)
        self.tabWidget.removeTab(self.tabWidget.indexOf(self.settings_tab))
        self.tabWidget.addTab(self.settings_tab, "Settings")

        # Align the tabs on the left and add the widget to the layout
        self.tabWidget.setTabPosition(QTabWidget.West)

        # Set active tab to the first one
        self.tabWidget.setCurrentIndex(0)

        # Obtain the Screen Width and Height as reference for the widget placement
        screen_size = QDesktopWidget().screenGeometry()

        # Create the Close Program button
        self.close_button = QPushButton("Close")

        # Hardcode the position to the bottom right of the screen
        self.close_button.setGeometry(QRect(round(0.92 * screen_size.width()), round(0.94 * screen_size.height()),
                                            round(0.05 * screen_size.width()), round(0.03 * screen_size.height())))

        # Close the program when the button is clicked
        self.close_button.clicked.connect(self.CloseApp)

        # Add the Close button to the main layout
        main_layout.addWidget(self.close_button, alignment=Qt.AlignRight)

        # Set the main widget and layout
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Define the CleanUp Thread
        self.cleanup_thread = None

    def CloseApp(self):
        # Show a dialog telling the user that the software is shutting down
        self.exiting_dialog = ExitingDialog(self)
        self.exiting_dialog.show()

        # Start the cleanup process in a separate thread
        self.cleanup_thread = CleanupThread(self.lane_tabs)

        # Link the Closing of the Dialog and the program upon finishing cleaning up the threads
        self.cleanup_thread.finished.connect(self.OnCleanUpFinished)

        # Start the clean up thread
        self.cleanup_thread.start()

    def OnCleanUpFinished(self):
        # Close the dialog and the application when cleanup is finished
        self.exiting_dialog.close()
        self.close()


class CleanupThread(QThread):
    finished = pyqtSignal()

    def __init__(self, lane_tabs):
        super().__init__()
        self.lane_tabs = lane_tabs

    def run(self):
        # Stop all activities in the lane tabs
        for lane_tab in self.lane_tabs:
            lane_tab.DisconnectSignalsAndStopRecorder()

        # Emit signal when finished
        self.finished.emit()


class ExitingDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        # Set a Window Title and Flags for the Exiting Dialog
        self.setWindowTitle("See you later!")
        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint)

        # Set its layout
        layout = QVBoxLayout(self)

        # Set a centered text and a loading gif for the exiting dialog
        label = QLabel("BowlEye Ball Tracker is shutting down", self)
        label.setAlignment(Qt.AlignCenter)
        self.loading_gif = QLabel(self)
        self.movie = QMovie("templates/loading.gif")
        self.movie.setScaledSize(QSize(120, 120))
        self.loading_gif.setAlignment(Qt.AlignCenter)
        self.loading_gif.setMovie(self.movie)
        self.movie.start()

        layout.addWidget(label)
        layout.addWidget(self.loading_gif)

        self.setLayout(layout)
        self.setModal(True)


if __name__ == "__main__":

    # Exception Handler
    def excepthook(exc_type, exc_value, exc_tb):
        error_message = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        QMessageBox.critical(None, "Unhandled Exception", error_message)

    sys.excepthook = excepthook
    # Function to show the GUI
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.showFullScreen()
    sys.exit(app.exec_())
