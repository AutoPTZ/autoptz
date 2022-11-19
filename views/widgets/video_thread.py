import time
import cv2
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import shared.constants as constants
import numpy as np


# test using QThread since frames will eventually be sent to QT
# if not use normal Python Threading

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0

    def __init__(self, src=0):
        super().__init__()
        self._run_flag = True
        self.cap = cv2.VideoCapture(src)
        (self.ret, self.frame) = self.cap.read()

    def run(self):
        while self._run_flag:
            (self.ret, self.frame) = self.cap.read()

            # FPS Counter
            self.fc += 1
            TIME = time.time() - self.start_time
            if TIME >= self.display_time:
                self.FPS = self.fc / TIME
                self.fc = 0
                self.start_time = time.time()
            fps = "FPS: " + str(self.FPS)[:5]
            cv2.putText(self.frame, fps, (50, 50), constants.FONT, 1, (0, 0, 255), 2)

            self.change_pixmap_signal.emit(self.frame)

        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
