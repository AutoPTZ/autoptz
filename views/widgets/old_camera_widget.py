from collections import deque
from threading import Thread, Lock
import time

import cv2
import imutils
from PyQt6 import QtCore, QtGui, QtWidgets

from logic.facial_tracking.dialogs.train_face import TrainerDlg
from logic.facial_tracking.old_image_processor import ImageProcessor


def start_trainer():
    TrainerDlg().show()


class CameraWidget(QtCore.QObject):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into frame
    """

    def __init__(self, width, height, camera_link=-1, aspect_ratio=False, parent=None, deque_size=1):
        super(CameraWidget, self).__init__(parent)

        # Initialize deque used to store frames read from the stream
        self.break_loop_lock = Lock()
        self.break_loop = False
        self.load_stream_thread = None
        self.deque = deque(maxlen=deque_size)

        # Slight offset is needed since PyQt layouts have a built-in padding
        # So add offset to counter the padding
        self.offset = 16
        self.screen_width = width - self.offset
        self.screen_height = height - self.offset
        self.maintain_aspect_ratio = aspect_ratio

        self.camera_stream_link = camera_link

        # Flag to check if camera is valid/working
        self.online = False
        self.capture = None
        self.video_frame = QtWidgets.QLabel()

        self.load_network_stream()

        # Start background frame grabbing
        self.get_frame_thread = Thread(target=self.get_frame, args=())
        self.get_frame_thread.daemon = True
        self.get_frame_thread.start()

        # Start Image Processor for Facial Recognition + Tracking
        self.image_processor = ImageProcessor()
        self.image_processor.START_TRAINER.connect(start_trainer)

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(1)

        print('Started camera: {}'.format(self.camera_stream_link))

    def load_network_stream(self):
        """Verifies stream link and open new stream if valid"""

        def load_network_stream_thread():
            if self.verify_network_stream(self.camera_stream_link):
                self.capture = cv2.VideoCapture(self.camera_stream_link)
                self.online = True

        self.load_stream_thread = Thread(target=load_network_stream_thread, args=())
        self.load_stream_thread.daemon = True
        self.load_stream_thread.start()

    def get_frame(self):
        """Reads frame, resizes, and converts image to pixmap"""

        while True:
            with self.break_loop_lock:
                if self.break_loop:
                    break
                else:
                    try:
                        timer = cv2.getTickCount()
                        if self.capture.isOpened() and self.online:
                            # Read next frame from stream and insert into deque
                            status, frame = self.capture.read()

                            # Keep frame aspect ratio
                            if self.maintain_aspect_ratio:
                                frame = imutils.resize(frame, width=self.screen_width)
                            # Force resize
                            else:
                                frame = cv2.resize(frame, (self.screen_width, self.screen_height))

                            frame = self.image_processor.get_frame(frame)

                            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                            frame = cv2.putText(frame, str(int(fps)), (75, 50), self.font, 0.7, (0, 0, 255),
                                                2)
                            if status:
                                self.deque.append(frame)
                            else:
                                self.capture.release()
                                self.online = False
                        else:
                            # Attempt to reconnect
                            print('attempting to reconnect', self.camera_stream_link)
                            self.load_network_stream()
                            self.spin(2)
                        self.spin(.01)
                    except AttributeError:
                        pass

    def set_frame(self):
        if self.break_loop:
            self.kill_video()
            return
        else:
            """Sets pixmap image to video frame"""
            if not self.online:
                self.spin(3)
                return

            if self.deque and self.online:
                # Grab latest frame
                frame = self.deque[-1]

                # Convert to pixmap and set to video frame
                img = QtGui.QImage(frame, frame.shape[1], frame.shape[0], frame.strides[0],
                                   QtGui.QImage.Format.Format_RGB888).rgbSwapped()
                try:
                    self.video_frame.setPixmap(QtGui.QPixmap.fromImage(img))
                except Exception as e:
                    print(e)
                    self.kill_video()

    @staticmethod
    def verify_network_stream(link):
        """Attempts to receive a frame from given link"""

        cap = cv2.VideoCapture(link)
        if not cap.isOpened():
            return False

        cap.release()
        return True

    @staticmethod
    def spin(seconds):
        """Pause for set amount of seconds, replaces time.sleep() so program doesn't stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

    def get_video_frame(self):
        return self.video_frame

    def kill_video(self):
        print("Killing Camera Object")

        with self.break_loop_lock:
            self.break_loop = True
        try:
            self.capture.release()
        except Exception as e:
            print(e)
        cv2.destroyAllWindows()
        self.load_stream_thread = None
        self.capture = None
        self.online = False
        print("Camera Object Done")
