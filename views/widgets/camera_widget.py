import math

from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
import cv2
import time
import dlib
import NDIlib as ndi
import shared.constants as constants
from logic.facial_tracking.dialogs.train_face import TrainerDlg
from logic.facial_tracking.image_processor import ImageProcessor
from views.widgets.video_thread import VideoThread


class CameraWidget(QLabel):
    """
    Create and handle all Cameras that are added to the UI.
    It creates a QLabel as OpenCV and NDI video can be converted to QPixmap for display.
    Combines both VideoThread and ImageProcessor threads for asynchronous computation for smoother looking video.
    With the latest frame, Dlib Object Tracking is in use and works alongside with FaceRecognition to fix any inconsistencies when tracking.
    """
    change_selection_signal = Signal()
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0
    stream_thread = None
    processor_thread = None
    track_started = None
    temp_tracked_name = None
    track_x = None
    track_y = None
    track_w = None
    track_h = None
    is_tracking = None
    tracked_name = None
    is_moving = False
    face_center_x = None
    face_center_y = None

    def __init__(self, source, width, height, lock, isNDI=False):
        super().__init__()
        self.width = width
        self.height = height
        self.lock = lock
        self.isNDI = isNDI
        self._is_stopped = True
        self.setProperty('active', False)
        if self.isNDI:
            self.setObjectName(f"Camera Source: {source.ndi_name}")
        else:
            self.setObjectName(f"Camera Source: {source}")
        self.setStyleSheet(constants.CAMERA_STYLESHEET)
        self.setText(f"Camera Source: {source}")
        self.mouseReleaseEvent = lambda event, widget=self: self.clicked_widget(
            event, widget)

        # Create Video Capture Thread
        self.stream_thread = VideoThread(src=source, width=width, isNDI=isNDI)
        # Connect it's Signal to the update_image Slot Method
        self.stream_thread.change_pixmap_signal.connect(self.update_image)
        # Start the Thread
        self.stream_thread.start()

        # Create and Run Image Processor Thread
        self.processor_thread = ImageProcessor(
            stream_thread=self.stream_thread, lock=self.lock)
        self.processor_thread.retrain_model_signal.connect(self.run_trainer)
        self.processor_thread.start()

        # PTZ Movement Thread
        self.last_request = None
        if isNDI and ndi.recv_ptz_is_supported(instance=self.stream_thread.ndi_recv):
            print(f"This NDI Source {source.ndi_name} Supports PTZ Movement")
            self.ptz_controller = self.stream_thread.ndi_recv
        else:
            if isNDI:
                print(
                    f"This NDI Source {source.ndi_name} Does NOT Supports PTZ Movement")
            self.ptz_controller = None
        self.ptz_is_usb = None

        self.track_started = False
        self.temp_tracked_name = None
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        self.is_tracking = False  # If Track Checkbox is checked
        self.tracked_name = None  # Face that needs to be tracked
        self.face_center_x = None
        self.face_center_y = None
        self.tracker = dlib.correlation_tracker()

    def stop(self):
        """
        When CameraWidget is being removed from the UI, we should stop associated PTZ cameras and all threads before deletion.
        """
        if self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                ndi.recv_ptz_pan_tilt_speed(
                    instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
        self.processor_thread.stop()
        self.stream_thread.stop()
        self.deleteLater()
        self.destroy()

    def set_ptz(self, control, isUSB=False):
        """
        Sets PTZ controller from Homepage
        If control is None then first stop PTZ and disconnect from it
        :param control:
        :param isUSB:
        """
        if control is None and self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                self.ptz_controller.pantilt(pan_speed=0, tilt_speed=0)
                self.ptz_controller.close_connection()
        self.ptz_controller = control
        self.ptz_is_usb = isUSB

    def set_add_name(self, name):
        """
        Run when the user wants to register a new face for recognition.
        If the Processor thread is already alive then just set the name to start taking images
        If the Processor thread is not alive then start up the thread, then add the face.
        :param name:
        """
        if self.processor_thread is not None:
            print("is running")
            self.processor_thread.add_name = name
        else:
            print(f"starting ImageProcessor Thread for {self.objectName()}")
            # Create and Run Image Processor Thread
            self.processor_thread = ImageProcessor(
                stream_thread=self.stream_thread, lock=self.lock)
            self.processor_thread.add_name = name
            self.processor_thread.start()

    def check_encodings(self):
        """
        Run when the user resets database or train a model.
        If the Processor thread is already alive then tell the thread to check encodings again.
        If the Processor thread is not alive then start up the thread, the thread will automatically check encodings.
        """
        if self.processor_thread is not None:
            self.processor_thread.check_encodings()
        else:
            print(f"starting ImageProcessor Thread for {self.objectName()}")
            self.processor_thread = ImageProcessor(
                stream_thread=self.stream_thread, lock=self.lock)
            self.processor_thread.start()

    def set_tracking(self):
        """
        Resets tracking variables when user toggles the checkbox
        """
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        self.is_tracking = not self.is_tracking
        if self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                ndi.recv_ptz_pan_tilt_speed(
                    instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
            self.last_request = None

    def set_tracked_name(self, name):
        """
        Sets the name to track, used to start tracking based on face recognition data
        :param name:
        """
        self.track_started = False
        self.tracked_name = name

    def get_tracking(self):
        """
        Returns if tracking is enabled
        :return:
        """
        return self.is_tracking

    def get_tracked_name(self):
        """
        Returns current tracked name
        :return:
        """
        return self.tracked_name

    def update_image(self, cv_img):
        """Updates the QLabel with the latest OpenCV/NDI frame and draws it"""
        cv_img = self.draw_on_frame(frame=cv_img)
        qt_img = self.convert_cv_qt(cv_img)
        self.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_qt_format.scaled(
            self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def clicked_widget(self, event, widget):
        """
        First checks if there is another CameraWidget currently active. If it is then deactivate it and update their stylesheet.
        Then if that deactivated CameraWidget is the same CameraWidget currently clicked on, then remove it from constants. So nothing is active.
        If it is not the same CameraWidget, then update this clicked on CameraWidget to be active and update its stylesheet.
        :param event:
        :param widget:
        """
        if constants.CURRENT_ACTIVE_CAM_WIDGET is not None:
            constants.CURRENT_ACTIVE_CAM_WIDGET.setProperty(
                'active', not constants.CURRENT_ACTIVE_CAM_WIDGET.property('active'))
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()

        if constants.CURRENT_ACTIVE_CAM_WIDGET == widget:
            constants.CURRENT_ACTIVE_CAM_WIDGET = None
        else:
            constants.CURRENT_ACTIVE_CAM_WIDGET = widget
            constants.CURRENT_ACTIVE_CAM_WIDGET.setProperty(
                'active', not constants.CURRENT_ACTIVE_CAM_WIDGET.property('active'))
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()
        self.change_selection_signal.emit()

    def draw_on_frame(self, frame):
        """
        Is called by update_image and returns the latest frame with FPS + face box drawings if there are any.
        :param frame:
        :return:
        """
        if self.processor_thread is not None:
            if self.processor_thread.face_locations is not None and self.processor_thread.face_names is not None and self.processor_thread.confidence_list is not None:
                for (top, right, bottom, left), name, confidence in zip(self.processor_thread.face_locations,
                                                                        self.processor_thread.face_names,
                                                                        self.processor_thread.confidence_list):

                    for box in self.processor_thread.body_locations:
                        (startX, startY, endX, endY) = box.astype("int")
                        cv2.rectangle(frame, (startX, startY),
                                      (endX, endY), (0, 255, 0), 2)
                        if name == self.tracked_name and left >= startX and top >= startY and right <= endX and bottom <= endY:
                            self.temp_tracked_name = name
                            self.track_x = startX
                            self.track_y = startY
                            self.track_w = endX
                            self.track_h = endY

                            # Calculate the center of the face
                            self.face_center_x = (left + right) // 2
                            self.face_center_y = (top + bottom) // 2

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 255, 0), 2)
                    # Draw a label with name and confidence for the face
                    cv2.putText(frame, name, (left + 5, top - 5),
                                constants.FONT, 0.5, (255, 255, 255), 1)
                    cv2.putText(frame, confidence, (right - 52, bottom - 5),
                                constants.FONT, 0.45, (255, 255, 0), 1)

        if self.is_tracking:
            frame = self.track_face(
                frame)
        self.temp_tracked_name = None

        # FPS Counter
        self.fc += 1
        time_set = time.time() - self.start_time
        if time_set >= self.display_time:
            self.FPS = self.fc / time_set
            self.fc = 0
            self.start_time = time.time()
        fps = "FPS: " + str(self.FPS)[:5]

        cv2.putText(frame, fps, (20, 30), constants.FONT, 0.7, (0, 0, 255), 2)
        return frame

    def track_face(self, frame):
        """
        Uses Dlib Object Tracking to set and update the currently tracked person
        Then if a PTZ camera is associated, it should move the camera in any direction automatically
        :param frame:
        :param x:
        :param y:
        :param w:
        :param h:
        :return:
        """
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        delta_x = 90  # Delta for left and right
        delta_y = 35  # Delta for up and down

        # Safe Zone
        cv2.ellipse(frame, (frame_center_x, frame_center_y), (delta_x, delta_y),
                    0, 0, 360, (0, 255, 0), 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, f"TRACKING {self.tracked_name.upper()}",
                    (20, 52), constants.FONT, 0.7, (0, 0, 255), 2)

        if not self.track_started or self.temp_tracked_name == self.tracked_name:
            rect = dlib.rectangle(self.track_x, self.track_y,
                                  self.track_w, self.track_h)
            self.tracker.start_track(rgb_frame, rect)
            self.track_started = True
        else:
            self.tracker.update(rgb_frame)
            pos = self.tracker.get_position()

            # unpack the position object
            self.track_x = int(pos.left())
            self.track_y = int(pos.top())
            self.track_w = int(pos.right())
            self.track_h = int(pos.bottom())

        # Calculate the center of the bounding box
        body_center_x = (self.track_x + self.track_w) // 2
        body_center_y = (self.track_y + self.track_h) // 2

        # If face is detected, calculate the center of the object as the average of the body center and face center
        if self.face_center_x is not None and self.face_center_y is not None:
            centerX = (body_center_x + self.face_center_x) // 2
            centerY = (body_center_y + self.face_center_y) // 1.75
            self.face_center_x = None
            self.face_center_y = None
        # If face is not detected, use the body center as the center of the object
        else:
            centerX = body_center_x
            centerY = body_center_y

        # Draw the center of the tracked object
        cv2.circle(frame, (centerX, int(centerY)), 5, (0, 0, 255), -1)

        cv2.putText(frame, "tracking", (self.track_x, self.track_h + 15),
                    constants.FONT, 0.45, (0, 255, 0), 1)
        cv2.rectangle(frame, (self.track_x, self.track_y),
                      (self.track_w, self.track_h), (255, 0, 255), 3, 1)

        # Draw the center of the frame
        cv2.circle(frame, (frame_center_x, frame_center_y),
                   5, (255, 0, 0), -1)

        # Draw the center of the tracked object
        cv2.circle(frame, (centerX, int(centerY)), 5, (0, 0, 255), -1)

        # Calculate the distance from the center
        distance_x = abs(centerX - frame_center_x)
        distance_y = abs(centerY - frame_center_y)

        # Calculate the maximum possible distance (from center to edge)
        max_distance_x = frame.shape[1] / 2
        max_distance_y = frame.shape[0] / 2

        # Normalize the distance (make it a value between 0 and 1)
        normalized_distance_x = distance_x / max_distance_x
        normalized_distance_y = distance_y / max_distance_y

        # Calculate the speed based on the normalized distance
        # The speed will be a value between 0.05 (for normalized_distance = 0) and 0.18 (for normalized_distance = 1)
        # Use a power function to make the speed increase more rapidly as the distance increases
        speed_x = 0.05 + (normalized_distance_x ** 3) * (0.2 - 0.05)
        speed_y = 0.05 + (normalized_distance_y ** 3) * (0.13 - 0.05)

        # If the object is within the delta range, set the speed to 0
        if abs(centerX - frame_center_x) <= delta_x:
            speed_x = 0
        if abs(centerY - frame_center_y) <= delta_y:
            speed_y = 0

        # Apply the direction to the speed
        if centerX > frame_center_x:
            speed_x = -speed_x
        if centerY > frame_center_y:
            speed_y = -speed_y

        if self.ptz_controller is not None:
            # Use centerX and centerY for PTZ control
            self.ptz_control(centerX, centerY, speed_x, speed_y,
                             frame_center_x, frame_center_y, delta_x, delta_y)

        return frame

    def ptz_control(self, centerX, centerY, speed_x, speed_y, frame_center_x, frame_center_y, delta_x, delta_y):
        # Define the directions
        directions = [("up_left", centerX < frame_center_x and centerY < frame_center_y),
                      ("down_left", centerX <
                       frame_center_x and centerY > frame_center_y),
                      ("left", centerX < frame_center_x),
                      ("up_right", centerX >
                       frame_center_x and centerY < frame_center_y),
                      ("down_right", centerX >
                       frame_center_x and centerY > frame_center_y),
                      ("right", centerX > frame_center_x),
                      ("up", centerY < frame_center_y),
                      ("down", centerY > frame_center_y),
                      ("stop", True)]

        for direction, condition in directions:
            if condition and self.last_request != direction:
                if self.ptz_is_usb:
                    getattr(self.ptz_controller, f"move_{direction}_track")()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=speed_x, tilt_speed=speed_y)
                self.last_request = direction
                break

    @staticmethod
    def run_trainer():
        """
        Runs Trainer on Main Thread
        """
        TrainerDlg().show()

    def closeEvent(self, event):
        """
        On event call, stop all the related threads.
        :param event:
        """
        if self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                self.ptz_controller.pantilt(pan_speed=0, tilt_speed=0)
                self.ptz_controller.close_connection()
        self.processor_thread.stop()
        self.stream_thread.stop()
        self.deleteLater()
        event.accept()
