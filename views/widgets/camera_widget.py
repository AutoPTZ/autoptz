from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
import cv2
import time
import imutils
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
        if self.processor_thread.is_alive():
            self.processor_thread.is_tracking = not self.processor_thread.is_tracking

    def set_tracked_name(self, name):
        print(f"Setting {name} as tracked name")
        if self.processor_thread.is_alive():
            self.processor_thread.tracked_name = name

    def get_tracked_name(self):
        if self.processor_thread.is_alive():
            return self.processor_thread.tracked_name

    def get_tracking(self):
        if self.processor_thread.is_alive():
            return self.processor_thread.is_tracking

    def update_image(self, cv_img):
        """Updates the QLabel with the latest OpenCV/NDI frame and draws it"""
        cv_img = self.draw_on_frame(frame=cv_img, face_locations=self.processor_thread.face_locations,
                                    face_names=self.processor_thread.face_names,
                                    confidence_list=self.processor_thread.confidence_list,
                                    track_x=self.processor_thread.track_x, track_y=self.processor_thread.track_y,
                                    track_w=self.processor_thread.track_w, track_h=self.processor_thread.track_h)
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

    def draw_on_frame(self, frame, face_locations, face_names, confidence_list, track_x, track_y, track_w, track_h):
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
                # print(top, right, bottom, left)
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with name and confidence for the face
                cv2.putText(frame, name, (left + 5, top - 5), constants.FONT, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, confidence, (right - 52, bottom - 5), constants.FONT, 0.45, (255, 255, 0), 1)

        if self.get_tracking():
            # min_x = int(frame.shape[1] / 11.5)
            # max_x = int(frame.shape[1] / 1.1)
            # min_y = int(frame.shape[0] / 8.5)
            # max_y = int(frame.shape[0] / 1.3)
            # cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
            if track_x is not None:
                cv2.putText(frame, "tracking", (track_x, track_h + 15), constants.FONT, 0.45, (0, 255, 0), 1)
                cv2.rectangle(frame, (track_x, track_y), (track_w, track_h), (255, 0, 255), 3, 1)

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

    def closeEvent(self, event):
        """
        On event call, stop all the related threads.
        :param event:
        """
        self.processor_thread.stop()
        self.stream_thread.stop()
        event.accept()
