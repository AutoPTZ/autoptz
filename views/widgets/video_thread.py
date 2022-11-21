import time
import cv2
from PySide6.QtCore import QThread, Signal
import shared.constants as constants
import numpy as np
import imutils


# test using QThread since frames will eventually be sent to QT
# if not use normal Python Threading

class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self, src, width):
        super().__init__()
        self._run_flag = True
        self.cap = cv2.VideoCapture(src)
        self.width = width
        # self.cv_img = None
        (self.ret, self.cv_img) = self.cap.read()

    def run(self):
        while self._run_flag:
            (self.ret, self.cv_img) = self.cap.read()
            if self.ret:
                # self.cv_img = imutils.resize(cv_img, width=self.width)
                self.change_pixmap_signal.emit(self.cv_img)

        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
