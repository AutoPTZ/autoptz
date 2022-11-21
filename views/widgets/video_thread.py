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

    # FPS for Performance
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0

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
                # FPS Counter
                self.fc += 1
                time_set = time.time() - self.start_time
                if time_set >= self.display_time:
                    self.FPS = self.fc / time_set
                    self.fc = 0
                    self.start_time = time.time()
                fps = "FPS: " + str(self.FPS)[:5]

                cv2.putText(self.cv_img, fps, (50, 50), constants.FONT, 1, (0, 0, 255), 2)

                self.change_pixmap_signal.emit(self.cv_img)

        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
