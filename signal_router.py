# import used packages
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtGui import QImage

# Create a class to store global signals
class SignalRouter(QObject):
    tracking_data_available = pyqtSignal()
    tracking_unsuccessful = pyqtSignal()
    pins_standing_signal = pyqtSignal(list)
    finished = pyqtSignal()
    debugging_image = pyqtSignal(QImage)

# Create a global instance of SignalRouter
signal_router = SignalRouter()
