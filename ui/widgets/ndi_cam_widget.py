import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from threading import Thread
from collections import deque
from datetime import datetime
import time
import sys
import cv2
import imutils
import NDIlib as ndi


class NDICameraWidget(QtWidgets.QWidget):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into fraame
    """

    def __init__(self, width, height, ndi_source=None, aspect_ratio=False, parent=None, deque_size=1):
        super(NDICameraWidget, self).__init__(parent)

        # Initialize deque used to store frames read from the stream
        self.deque = deque(maxlen=deque_size)

        # Slight offset is needed since PyQt layouts have a built in padding
        # So add offset to counter the padding
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio

        self.ndi_source_object = ndi_source
        self.ndi_recv_create = ndi.RecvCreateV3()
        self.ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        self.ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_LOWEST
        self.ndi_recv = ndi.recv_create_v3(self.ndi_recv_create)

        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.video_frame = QtWidgets.QLabel()

        self.load_network_stream()

        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(1)

        print('Started camera: {}'.format(self.ndi_source_object.ndi_name))

    def load_network_stream(self):
        """Verifies NDI source and open stream if valid"""

        def load_network_stream_thread():
            if self.ndi_recv is None:
                return 0
            ndi.recv_connect(self.ndi_recv, self.ndi_source_object)
            self.online = True

        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""
        while True:
            timer = cv2.getTickCount()
            try:
                t, v, _, _ = ndi.recv_capture_v3(self.ndi_recv, 5000)
                if self.online:
                    # Read next frame from stream and insert into deque
                    try:
                        if t == ndi.FRAME_TYPE_VIDEO:

                            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                            print(fps, self.ndi_source_object.ndi_name)
                            frame = np.copy(v.data)
                            frame = cv2.putText(frame, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                     (0, 0, 255), 2)
                        else:
                            frame = np.copy(v.data)

                        self.deque.append(frame)
                    except:
                         self.online = False
                else:
                    # Attempt to reconnect
                    print('attempting to reconnect', self.ndi_source_object.ndi_name)
                    self.load_network_stream()
                    self.spin(2)
                self.spin(.001)
            except AttributeError:
                pass

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

    def set_frame(self):
        """Sets pixmap image to video frame"""

        if not self.online:
            self.spin(1)
            return

        if self.deque and self.online:
            # Grab latest frame
            frame = self.deque[-1]

            # Keep frame aspect ratio
            if self.maintain_aspect_ratio:
                self.frame = frame
                # self.frame = imutils.resize(frame, width=self.screen_width)
            # Force resize
            else:
                self.frame = frame
                # self.frame = cv2.resize(frame, (self.screen_width, self.screen_height))

            # Convert to pixmap and set to video frame
            self.img = QtGui.QImage(self.frame, self.frame.shape[1], self.frame.shape[0], self.frame.strides[0],
                                    QtGui.QImage.Format_RGBX8888).rgbSwapped()

            try:
                self.video_frame.setPixmap(QtGui.QPixmap.fromImage(self.img))
            except:
                print("Killing Camera Object")
                ndi.recv_destroy(self.ndi_recv)
                ndi.destroy()
                self.online = False
                self.capture = None
                cv2.destroyAllWindows()

    def get_video_frame(self):
        return self.video_frame

    def kill_video(self):
        print("Killing Camera Object")
        ndi.recv_destroy(self.ndi_recv)
        ndi.destroy()
        self.online = False
        self.capture = None
        cv2.destroyAllWindows()