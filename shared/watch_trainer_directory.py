import time

import watchdog.events
import watchdog.observers
from PySide6 import QtWidgets


class WatchTrainer(watchdog.events.PatternMatchingEventHandler):

    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['*.pickle'],
                                                             ignore_directories=True, case_sensitive=False)
        self.camera_widget_list = []

    def add_camera(self, camera_widget):
        self.camera_widget_list.append(camera_widget)

    def remove_camera(self, camera_widget):
        self.camera_widget_list.remove(camera_widget)

    def on_any_event(self, event):
        print("Watchdog received an event at - % s." % event.src_path)
        self.spin(1)
        for camera in self.camera_widget_list:
            camera.processor_thread.check_encodings()

    @staticmethod
    def spin(seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()
