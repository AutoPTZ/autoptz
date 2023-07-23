import time
import watchdog.events
import watchdog.observers
from PySide6 import QtWidgets


class WatchTrainer(watchdog.events.PatternMatchingEventHandler):
    """
    WatchTrainer is used to tell all currently active Cameras to reload the encoded faces file, if any.
    """

    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(
            self, patterns=['*.pickle'], ignore_directories=True)
        self.camera_widget_list = []

    def add_camera(self, camera_widget):
        """
        Add newly CameraWidget to the WatchTrainer list
        :param camera_widget:
        """
        self.camera_widget_list.append(camera_widget)

    def remove_camera(self, camera_widget):
        """
        Removes recently deleted CameraWidget from WatchTrainer list
        :param camera_widget:
        """
        self.camera_widget_list.remove(camera_widget)

    def on_created(self, event):
        """
        Only when encodings file is created, then tell all the cameras sources to load the file
        :param event:
        """
        print("Watchdog received an event at - % s." % event.src_path)
        self.spin(5)
        for camera in self.camera_widget_list:
            camera.facial_recognition.check_encodings()

    def on_deleted(self, event):
        """
        Only when encodings file is deleted, then tell all the cameras sources to forget encoded data
        :param event:
        """
        print("Watchdog received an event at - % s." % event.src_path)
        self.spin(5)
        for camera in self.camera_widget_list:
            camera.facial_recognition.check_encodings()

    def on_modified(self, event):
        """
        Only when encodings file is modified, then tell all the cameras sources to refresh encoded data
        :param event:
        """
        print("Watchdog received an event at - % s." % event.src_path)
        self.spin(5)
        for camera in self.camera_widget_list:
            camera.facial_recognition.check_encodings()

    @staticmethod
    def spin(seconds):
        """Pause for set amount of seconds, replaces time.sleep() so program doesn't stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()
