import watchdog.events
import watchdog.observers


class WatchTrainer(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['*.txt', '*.yml'],
                                                             ignore_directories=True, case_sensitive=False)
        self.camera_widget_list = []

    def add_camera(self, camera):
        self.camera_widget_list.append(camera)

    def remove_camera(self, camera):
        self.camera_widget_list.remove(camera)

    def on_any_event(self, event):
        print("Watchdog received an event at - % s." % event.src_path)
        for camera in self.camera_widget_list:
            camera.resetFacialRecognition()
