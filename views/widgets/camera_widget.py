from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
import cv2

import shared.constants as constants
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
        # create the video capture thread
        self.thread = VideoThread(src=source)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()


    def stop(self):
        self.thread.stop()
        self.deleteLater()

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)
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

    def draw_recognized_face(self, frame, face_locations, face_names, confidence_list=None):

        if confidence_list is None:
            confidence_list = [0]
        for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidence_list):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with name and confidence for the face
            cv2.putText(frame, name, (left + 5, top - 5), self.font, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, confidence, (right - 52, bottom - 5), self.font, 0.45, (255, 255, 0), 1)

        return frame

    def closeEvent(self, event):
        self.thread.stop()
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
