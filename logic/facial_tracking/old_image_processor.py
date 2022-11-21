import os
import cv2
from PyQt6 import QtCore
import dlib
import shared.constants as constants
from logic.facial_tracking.old_track_handler import TrackHandler


class ImageProcessor(QtCore.QObject):
    START_TRAINER = QtCore.pyqtSignal()

    def __init__(self):
        super(ImageProcessor, self).__init__()

        # Facial Recognition & Object Tracking
        self.is_adding_face = False
        self.adding_to_name = None
        self.count = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.name_id = None
        self.enable_track_checked = False
        self.tracked_name = None
        self.track_started = None
        self.tracker = None
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        self.track_handler = TrackHandler()

        # VISCA/ONVIF PTZ Control
        self.ptz_ready = None
        self.camera_control = None

    def get_frame(self, frame):
        if self.is_adding_face:
            return self.add_face(frame)
        try:
            if os.path.exists(constants.ENCODINGS_PATH):
                face_locations, face_names, confidence_list = self.track_handler.recognize_face(frame)
                frame = self.draw_recognized_face(frame, face_locations, face_names, confidence_list)
                # frame = self.track_handler.yolo_detector(frame)
                # frame = self.track_handler.yolo_detector_faster(frame)
                # frame = self.track_handler.mobile_ssd_detector(frame)
        except Exception as e:
            print(e)
            return frame
        if self.enable_track_checked and self.track_x is not None and self.track_y is not None and self.track_w is not None and self.track_h is not None:
            frame = self.track_face(frame, self.track_x, self.track_y, self.track_w, self.track_h)
        self.name_id = None
        return frame

    def track_face(self, frame, x, y, w, h):
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, "Tracking Enabled", (75, 75), self.font, 0.7, (0, 0, 255), 2)
        min_x = int(frame.shape[1] / 11.5)
        max_x = int(frame.shape[1] / 1.1)
        min_y = int(frame.shape[0] / 8.5)
        max_y = int(frame.shape[0] / 1.3)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        if not self.track_started:
            self.tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, w, h)
            self.tracker.start_track(rgbFrame, rect)
            self.track_started = True
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 255), 3, 1)
        if self.name_id == self.tracked_name:
            rect = dlib.rectangle(x, y, w, h)
            self.tracker.start_track(rgbFrame, rect)
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 255), 3, 1)
            cv2.putText(frame, "tracking", (x, h + 15), self.font, 0.45, (0, 255, 0), 1)
        else:
            self.tracker.update(rgbFrame)
            pos = self.tracker.get_position()
            # unpack the position object
            x = int(pos.left())
            y = int(pos.top())
            w = int(pos.right())
            h = int(pos.bottom())
            cv2.rectangle(frame, (x - 5, y - 5), (w + 5, h + 5), (255, 0, 255), 3, 1)
            cv2.putText(frame, "tracking", (x, h + 20), self.font, 0.45, (0, 255, 0), 1)

        if self.camera_control is not None:
            if self.ptz_ready is None:
                # For VISCA PTZ
                if x > min_x and w < max_x and y > min_y and h < max_y:
                    self.camera_control.move_stop()
                if w > max_x:
                    self.camera_control.move_right_track()
                elif x < min_x:
                    self.camera_control.move_left_track()
                if h > max_y:
                    self.camera_control.move_down_track()
                elif y < min_y:
                    self.camera_control.move_up_track()
            else:
                # For ONVIF PTZ
                if x > min_x and w < max_x and y > min_y and h < max_y:
                    self.camera_control.stop_move()
                    # movementX = False
                    # faster_movement = False
                if w > max_x:
                    self.camera_control.continuous_move(0.05, 0, 0)
                    # movementX = False
                elif x < min_x:
                    self.camera_control.continuous_move(-0.05, 0, 0)
                    # movementX = False
                if h > min_y:
                    self.camera_control.continuous_move(0, -0.05, 0)
                    # movementY = False
                elif y < max_y:
                    self.camera_control.continuous_move(0, 0.05, 0)
                    # movementY = False
        return frame

    def set_face(self, name):
        self.tracked_name = name

    def get_face(self):
        return self.tracked_name

    def set_ptz_ready(self, text):
        self.ptz_ready = text

    def get_ptz_ready(self):
        return self.ptz_ready

    def config_enable_track(self):
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        if self.camera_control is not None:
            if self.ptz_ready is None:
                self.camera_control.move_stop()
            else:
                self.camera_control.stop_move()
        self.enable_track_checked = not self.enable_track_checked

    def is_track_enabled(self):
        return self.enable_track_checked

    def set_ptz_controller(self, control):
        self.camera_control = control
