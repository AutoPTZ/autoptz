import time

import watchdog.events
import watchdog.observers
from PyQt5 import QtWidgets


class WatchTrainer(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['*.yml'],
                                                             ignore_directories=True, case_sensitive=False)
        self.camera_widget_list = []

    def add_camera(self, camera):
        self.camera_widget_list.append(camera)

    def remove_camera(self, camera):
        self.camera_widget_list.remove(camera)

    def on_any_event(self, event):
        print("Watchdog received an event at - % s." % event.src_path)
        print("Will reconfigure all inuse camera facial recognition")
        self.spin(15)
        for camera in self.camera_widget_list:
            camera.resetFacialRecognition()

    @staticmethod
    def spin(seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()