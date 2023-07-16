import os
import cv2
from PySide6.QtCore import QThread, Signal
import numpy as np
import NDIlib as ndi
import imutils


class VideoThread(QThread):
    """
    Threaded Video Capture for both OpenCV and NDI camera feeds.
    Emits/Returns the latest frame to the relative CameraWidget.
    """
    change_pixmap_signal = Signal(np.ndarray)
    _run_flag = None
    resize_width = None
    isNDI = None
    ndi_recv = None
    ret = None
    cv_img = None
    imutils = None

    def __init__(self, src, width, isNDI=False):
        super().__init__()
        self._run_flag = True
        self.resize_width = width
        self.daemon = True
        self.isNDI = isNDI
        self.imutils = imutils
        self.src = src
        if type(src) == int:
            self.cap = cv2.VideoCapture(src)
            if os.name == 'nt':  # fixes Windows OpenCV resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)
            (self.ret, img) = self.cap.read()
            self.cv_img = self.imutils.resize(img, width=self.resize_width)
        else:
            ndi_recv_create = ndi.RecvCreateV3()
            ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            # ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_LOWEST
            self.ndi_recv = ndi.recv_create_v3(ndi_recv_create)
            ndi.recv_connect(self.ndi_recv, src)
            self.ret, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
            if self.ret == ndi.FRAME_TYPE_VIDEO:
                cv_img = np.copy(v.data)
                ndi.recv_free_video_v2(self.ndi_recv, v)
                self.cv_img = imutils.resize(cv_img, width=self.resize_width)


    def run(self):
        """
        Runs continuously on CameraWidget.start() to provide the latest video frame until _run_flag is False.
        """
        while self._run_flag:
            if type(self.src) == int:
                (self.ret, img) = self.cap.read()
                if self.ret:
                    self.cv_img = self.imutils.resize(img, width=self.resize_width)
                    self.change_pixmap_signal.emit(self.cv_img)
            else:
                self.ret, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
                if self.ret == ndi.FRAME_TYPE_VIDEO:
                    cv_img = np.copy(v.data)
                    ndi.recv_free_video_v2(self.ndi_recv, v)
                    self.cv_img = imutils.resize(cv_img, width=self.resize_width)
                    self.change_pixmap_signal.emit(self.cv_img)

        # shut down capture system
        if self.isNDI:
            ndi.recv_destroy(self.ndi_recv)
            ndi.destroy()
        else:
            self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
        self.deleteLater()
