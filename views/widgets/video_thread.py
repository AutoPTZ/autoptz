import time
import cv2
from PySide6.QtCore import QThread, Signal
import shared.constants as constants
import numpy as np
import NDIlib as ndi
import imutils


# test using QThread since frames will eventually be sent to QT
# if not use normal Python Threading

class VideoThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self, src, width, isNDI=False):
        super().__init__()
        self._run_flag = True
        self.width = width
        self.isNDI = isNDI
        if isNDI:
            ndi_source_object = src
            ndi_recv_create = ndi.RecvCreateV3(ndi_source_object)
            ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_MAX
            self.ndi_recv = ndi.recv_create_v3(ndi_recv_create)
            self.ret, v, _, _ = ndi.recv_capture_v3(self.ndi_recv, 5000)
        else:
            self.cap = cv2.VideoCapture(src)
            (self.ret, self.cv_img) = self.cap.read()

    def run(self):
        while self._run_flag:
            if self.isNDI:
                self.ret, v, _, _ = ndi.recv_capture_v3(self.ndi_recv, 5000)
                if self.ret == ndi.FRAME_TYPE_VIDEO:
                    self.cv_img = np.copy(v.data)
                    # self.cv_img = imutils.resize(cv_img, width=self.width)
                    self.change_pixmap_signal.emit(self.cv_img)
            else:
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
