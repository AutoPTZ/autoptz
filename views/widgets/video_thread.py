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
    width = None
    isNDI = None
    ndi_recv = None
    ret = None
    cv_img = None
    imutils = None

    def __init__(self, src, width, isNDI=False):
        super().__init__()
        self._run_flag = True
        self.width = width
        self.daemon = True
        self.isNDI = isNDI
        self.imutils = imutils
        if isNDI:
            ndi_recv_create = ndi.RecvCreateV3()
            ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
            # ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_MAX
            self.ndi_recv = ndi.recv_create_v3(ndi_recv_create)
            ndi.recv_connect(self.ndi_recv, src)
            self.ret, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
            if self.ret == ndi.FRAME_TYPE_VIDEO:
                self.cv_img = np.copy(v.data)
                ndi.recv_free_video_v2(self.ndi_recv, v)
                # v = np.copy(v.data)
                # self.cv_img = imutils.resize(v, width=self.width)

        else:
            self.cap = cv2.VideoCapture(src)
            (self.ret, img) = self.cap.read()
            self.cv_img = self.imutils.resize(img, width=self.width)

    def run(self):
        """
        Runs continuously on CameraWidget.start() to provide the latest video frame until _run_flag is False.
        """
        while self._run_flag:
            if self.isNDI:
                self.ret, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
                if self.ret == ndi.FRAME_TYPE_VIDEO:
                    self.cv_img = np.copy(v.data)
                    # v = np.copy(v.data)
                    # self.cv_img = imutils.resize(v, width=self.width)
                    self.change_pixmap_signal.emit(self.cv_img)
                    ndi.recv_free_video_v2(self.ndi_recv, v)
            else:
                (self.ret, img) = self.cap.read()
                if self.ret:
                    self.cv_img = self.imutils.resize(img, width=self.width)
                    self.change_pixmap_signal.emit(self.cv_img)

        # shut down capture system
        self.cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
