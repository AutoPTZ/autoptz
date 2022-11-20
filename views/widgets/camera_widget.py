from PyQt6 import QtGui
from PyQt6.QtWidgets import QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import pyqtSlot, Qt
import cv2
import numpy as np

import shared.constants as constants
from views.widgets.video_thread import VideoThread


# For now this only creates USB Camera Widget
class CameraWidget(QLabel):

    def __init__(self, source, width, height):
        super().__init__()

        self.width = width
        self.height = height
        self.setProperty('active', False)
        self.resize(width, height)
        self.setObjectName(f"Camera Source: {source}")
        self.setStyleSheet(constants.CAMERA_STYLESHEET)
        self.setText(f"Camera Source: {source}")

        # create the video capture thread
        self.thread = VideoThread(src=source)
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()

    def stop(self):
        self.thread.stop()
        self.deleteLater()

    @pyqtSlot(np.ndarray)
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


