from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
import cv2

import shared.constants as constants
from logic.facial_tracking.testing_image_processor import ImageProcessor
from views.widgets.video_thread import VideoThread


# For now this only creates USB Camera Widget
class CameraWidget(QLabel):
    change_selection_signal = Signal(bool)

    def __init__(self, source, width, height):
        super().__init__()
        self.width = width
        self.height = height
        self.setProperty('active', False)
        self.resize(width, height)
        self.setObjectName(f"Camera Source: {source}")
        self.setStyleSheet(constants.CAMERA_STYLESHEET)
        self.setText(f"Camera Source: {source}")
        self.mouseReleaseEvent = lambda event, widget=self: self.clicked_widget(event, widget)

        # Create Video Capture Thread
        self.stream_thread = VideoThread(src=source)
        # Connect it's Signal to the update_image Slot Method
        self.stream_thread.change_pixmap_signal.connect(self.update_image)
        # Start the Thread
        self.stream_thread.start()

        # Create and Run Image Processor Thread
        self.processor_thread = ImageProcessor(stream_thread=self.stream_thread).start()

    def stop(self):
        self.stream_thread.stop()
        self.deleteLater()

    def set_add_name(self, name):
        self.processor_thread.add_name = name

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image and draws on latest frame if processing is completed"""
        cv_img = self.draw_on_face(frame=cv_img, face_locations=self.processor_thread.face_locations, face_names=self.processor_thread.face_names, confidence_list=self.processor_thread.confidence_list)
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
        self.change_selection_signal.emit(True)

    def draw_on_face(self, frame, face_locations, face_names, confidence_list=None):
        if face_locations is not None:
            if confidence_list is None:
                confidence_list = [0]
            for (top_left, bottom_left, top_right, bottom_right), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top_left *= 2
                bottom_left *= 2
                top_right *= 2
                bottom_right *= 2

                # Draw a box around the face
                cv2.rectangle(frame, (top_left, bottom_left), (top_right, bottom_right), (0, 255, 0), 3)

                # Draw a label with name and confidence for the face
                cv2.putText(frame, name, (top_left + 5, bottom_left - 5), constants.FONT, 1, (255, 255, 255), 1)
                # cv2.putText(frame, confidence, (right - 52, bottom - 5), self.font, 0.45, (255, 255, 0), 1)

        return frame

    def closeEvent(self, event):
        self.processor_thread.stop()
        self.stream_thread.stop()
        event.accept()

    """
             OpenCV VideoThread -> sent for processing (which easily uses OpenCV frame)

             QT uses QPixmaps, so we need to convert OpenCV frame to QPixmap

             FrameProcess == Facial Recognition, Tracking, etc
             VideoThread == Turns on Camera and constantly gets frams
             CameraWidget == Shows the frames on QT to the user

             VideoThread -> FrameProcess (sends back, boxes + names)

             VideoThread -> Pixmap -> CameraWidget
             VideoThread -> CameraWidget -> Pixmap


             VideoThread ->  CameraWidget  <- FrameProcess (boxes)
             CameraWidget -> Pixmap

            """
