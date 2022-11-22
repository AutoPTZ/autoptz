from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
import cv2
import time
import imutils
import dlib
import shared.constants as constants
from logic.facial_tracking.testing_image_processor import ImageProcessor
from views.widgets.video_thread import VideoThread


class CameraWidget(QLabel):
    """
    Create and handle all Cameras that are added to the UI.
    It creates a QLabel as OpenCV and NDI video can be converted to QPixmap for display.
    Combines both VideoThread and ImageProcessor threads for asynchronous computation for smoother looking video.
    """
    change_selection_signal = Signal()
    # FPS for Performance
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0

    def __init__(self, source, width, height, isNDI=False):
        super().__init__()
        self.width = width
        self.height = height
        self.setProperty('active', False)
        # self.resize(width, height)
        self.setObjectName(f"Camera Source: {source}")
        self.setStyleSheet(constants.CAMERA_STYLESHEET)
        self.setText(f"Camera Source: {source}")
        self.mouseReleaseEvent = lambda event, widget=self: self.clicked_widget(event, widget)

        # Create Video Capture Thread
        self.stream_thread = VideoThread(src=source, width=width, isNDI=isNDI)
        # Connect it's Signal to the update_image Slot Method
        self.stream_thread.change_pixmap_signal.connect(self.update_image)
        # Start the Thread
        self.stream_thread.start()

        # Create and Run Image Processor Thread
        self.processor_thread = ImageProcessor(stream_thread=self.stream_thread)
        self.processor_thread.start()

        self.tracker = None
        self.temp_tracked_name = None
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        self.is_tracking = False  # If Track Checkbox is checked
        self.tracked_name = None  # Face that needs to be tracked

    def stop(self):
        """
        When CameraWidget is being removed from the UI, we should stop all relevant threads before deletion.
        """
        self.processor_thread.stop()
        self.stream_thread.stop()
        self.deleteLater()

    def set_add_name(self, name):
        """
        Run when the user wants to register a new face for recognition.
        If the Processor thread is already alive then just set the name to start taking images,
        If the Processor thread is not alive then start up the thread, then add the face.
        :param name:
        """
        if self.processor_thread.is_alive():
            self.processor_thread.add_name = name
        else:
            print(f"starting ImageProcessor Thread for {self.objectName()}")
            # Create and Run Image Processor Thread
            self.processor_thread = ImageProcessor(stream_thread=self.stream_thread)
            self.processor_thread.add_name = name
            self.processor_thread.start()

    def check_encodings(self):
        """
        Run when the user resets database or train a model.
        If the Processor thread is already alive then tell the thread to check encodings again.
        If the Processor thread is not alive then start up the thread, the thread will automatically check encodings.
        """
        if self.processor_thread.is_alive():
            self.processor_thread.check_encodings()
        else:
            print(f"starting ImageProcessor Thread for {self.objectName()}")
            self.processor_thread = ImageProcessor(stream_thread=self.stream_thread)
            self.processor_thread.start()

    def set_tracking(self):
        self.is_tracking = not self.is_tracking

    def set_tracked_name(self, name):
        self.tracked_name = name

    def get_tracked_name(self):
        return self.tracked_name

    def get_tracking(self):
        return self.is_tracking

    def update_image(self, cv_img):
        """Updates the QLabel with the latest OpenCV/NDI frame and draws it"""
        cv_img = self.draw_on_frame(frame=cv_img, face_locations=self.processor_thread.face_locations,
                                    face_names=self.processor_thread.face_names,
                                    confidence_list=self.processor_thread.confidence_list)
        qt_img = self.convert_cv_qt(cv_img)
        self.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_qt_format.scaled(self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)
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
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()

        if constants.CURRENT_ACTIVE_CAM_WIDGET == widget:
            constants.CURRENT_ACTIVE_CAM_WIDGET = None
        else:
            constants.CURRENT_ACTIVE_CAM_WIDGET = widget
            constants.CURRENT_ACTIVE_CAM_WIDGET.setProperty(
                'active', not constants.CURRENT_ACTIVE_CAM_WIDGET.property('active'))
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()
        self.change_selection_signal.emit()

    def draw_on_frame(self, frame, face_locations, face_names, confidence_list):
        """
        Is called by update_image and returns the latest frame with FPS + face box drawings if there are any.
        :param frame:
        :param face_locations:
        :param face_names:
        :param confidence_list:
        :return:
        """
        if face_locations is not None and face_names is not None and confidence_list is not None:
            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidence_list):
                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                if name == self.tracked_name:
                    self.temp_tracked_name = name
                    self.track_x = left
                    self.track_y = top
                    self.track_w = right
                    self.track_h = bottom
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with name and confidence for the face
                cv2.putText(frame, name, (left + 5, top - 5), constants.FONT, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, confidence, (right - 52, bottom - 5), constants.FONT, 0.45, (255, 255, 0), 1)

        if self.is_tracking and self.processor_thread.is_alive():
            frame = self.track_face(frame)

        # FPS Counter
        self.fc += 1
        time_set = time.time() - self.start_time
        if time_set >= self.display_time:
            self.FPS = self.fc / time_set
            self.fc = 0
            self.start_time = time.time()
        fps = "FPS: " + str(self.FPS)[:5]

        cv2.putText(frame, fps, (50, 50), constants.FONT, 1, (0, 0, 255), 2)
        return frame

    def track_face(self, frame):  # Probably needs to be on its own thread
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.tracker is None and self.track_x is not None:
            self.tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(self.track_x, self.track_y, self.track_w, self.track_h)
            self.tracker.start_track(rgb_frame, rect)
            cv2.putText(frame, "tracking", (self.track_x, self.track_h + 15), constants.FONT, 0.45, (0, 255, 0), 1)
            cv2.rectangle(frame, (self.track_x, self.track_y), (self.track_w, self.track_h), (255, 0, 255), 3, 1)
        if self.tracker is not None and self.temp_tracked_name == self.tracked_name and self.track_x is not None:
            rect = dlib.rectangle(self.track_x, self.track_y, self.track_w, self.track_h)
            self.tracker.start_track(rgb_frame, rect)
            cv2.putText(frame, "tracking", (self.track_x, self.track_h + 15), constants.FONT, 0.45, (0, 255, 0), 1)
            cv2.rectangle(frame, (self.track_x, self.track_y), (self.track_w, self.track_h), (255, 0, 255), 3, 1)
        elif self.tracker is not None:
            self.tracker.update(rgb_frame)
            pos = self.tracker.get_position()
            # unpack the position object
            self.track_x = int(pos.left())
            self.track_y = int(pos.top())
            self.track_w = int(pos.right())
            self.track_h = int(pos.bottom())
            cv2.putText(frame, "tracking", (self.track_x, self.track_h + 15), constants.FONT, 0.45, (0, 255, 0), 1)
            cv2.rectangle(frame, (self.track_x, self.track_y), (self.track_w, self.track_h), (255, 0, 255), 3, 1)

        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        self.temp_tracked_name = None
        return frame

    def closeEvent(self, event):
        """
        On event call, stop all the related threads.
        :param event:
        """
        self.processor_thread.stop()
        self.stream_thread.stop()
        event.accept()
