import os
import time
import cv2
from PySide6.QtCore import QThread, Signal
import numpy as np
import NDIlib as ndi
import imutils

from multiprocessing import Process, Queue


class VideoThread(QThread):
    """
    Threaded Video Capture for both OpenCV and NDI camera feeds.
    Emits/Returns the latest frame to the relative CameraWidget.
    """
    change_pixmap_signal = Signal(np.ndarray)

    def __init__(self, shm, lock, src, width, isNDI=False):
        super().__init__()
        self._run_flag = True
        self.resize_width = width
        self.daemon = True
        self.isNDI = isNDI
        self.src = src
        self.shm = shm
        self.lock = lock

    def run(self):
        while self._run_flag:
            # Access the shared memory block
            with self.lock:
                shm_buf = np.ndarray(
                    (self.resize_width, self.resize_width, 3), dtype=np.uint8, buffer=self.shm.buf)
                if np.any(shm_buf):  # If the frame is not empty
                    self.change_pixmap_signal.emit(shm_buf)
                else:
                    time.sleep(0.01)  # wait for 10 milliseconds

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
        self.deleteLater()


def run_process(frame_queue, src, width, isNDI=False):
    _run_flag = True
    resize_width = width
    if type(src) == int:
        cap = cv2.VideoCapture(src)
        if os.name == 'nt':  # fixes Windows OpenCV resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)
    else:
        ndi_recv_create = ndi.RecvCreateV3()
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        ndi_recv = ndi.recv_create_v3(ndi_recv_create)
        ndi.recv_connect(ndi_recv, src)
    while _run_flag:
        if type(src) == int:
            ret, img = cap.read()
            if ret:
                cv_img = imutils.resize(img, width=resize_width)
                frame_queue.put(cv_img)
        else:
            ret, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)
            if ret == ndi.FRAME_TYPE_VIDEO:
                cv_img = np.copy(v.data)
                ndi.recv_free_video_v2(ndi_recv, v)
                cv_img = imutils.resize(cv_img, width=resize_width)
                frame_queue.put(cv_img)
    if isNDI:
        ndi.recv_destroy(ndi_recv)
        ndi.destroy()
    else:
        cap.release()


# class VideoThread(QThread):

#     def __init__(self, src, width, isNDI=False):
#         super().__init__()
#         self._run_flag = True
#         self.resize_width = width
    #     self.daemon = True
    #     self.isNDI = isNDI
    #     self.imutils = imutils
    #     self.src = src
    #     if type(src) == int:
    #         self.cap = cv2.VideoCapture(src)
    #         if os.name == 'nt':  # fixes Windows OpenCV resolution
    #             self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)
    #             self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)
    #         (self.ret, img) = self.cap.read()
    #         self.cv_img = self.imutils.resize(img, width=self.resize_width)
    #     else:
    #         ndi_recv_create = ndi.RecvCreateV3()
    #         ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
    #         # ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_LOWEST
    #         self.ndi_recv = ndi.recv_create_v3(ndi_recv_create)
    #         ndi.recv_connect(self.ndi_recv, src)
    #         self.ret, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
    #         if self.ret == ndi.FRAME_TYPE_VIDEO:
    #             cv_img = np.copy(v.data)
    #             ndi.recv_free_video_v2(self.ndi_recv, v)
    #             self.cv_img = imutils.resize(cv_img, width=self.resize_width)

    # def run(self):
    #     """
    #     Runs continuously on CameraWidget.start() to provide the latest video frame until _run_flag is False.
    #     """
    #     while self._run_flag:
    #         if type(self.src) == int:
    #             (self.ret, img) = self.cap.read()
    #             if self.ret:
    #                 self.cv_img = self.imutils.resize(
    #                     img, width=self.resize_width)
    #                 self.change_pixmap_signal.emit(self.cv_img)
    #         else:
    #             self.ret, v, _, _ = ndi.recv_capture_v2(self.ndi_recv, 5000)
    #             if self.ret == ndi.FRAME_TYPE_VIDEO:
    #                 cv_img = np.copy(v.data)
    #                 ndi.recv_free_video_v2(self.ndi_recv, v)
    #                 self.cv_img = imutils.resize(
    #                     cv_img, width=self.resize_width)
    #                 self.change_pixmap_signal.emit(self.cv_img)

    #     # shut down capture system
    #     if self.isNDI:
    #         ndi.recv_destroy(self.ndi_recv)
    #         ndi.destroy()
    #     else:
    #         self.cap.release()

    # def stop(self):
    #     """Sets run flag to False and waits for thread to finish"""
    #     self._run_flag = False
    #     self.wait()
    #     self.deleteLater()
