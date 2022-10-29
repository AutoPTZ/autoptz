import os.path
from collections import deque
from threading import Thread, Lock
import time

import cv2
import imutils
from PyQt5 import QtCore, QtGui, QtWidgets

from logic.facial_tracking.train_face import Trainer


class CameraWidget(QtWidgets.QWidget):
    """Independent camera feed
    Uses threading to grab IP camera frames in the background

    @param width - Width of the video frame
    @param height - Height of the video frame
    @param stream_link - IP/RTSP/Webcam link
    @param aspect_ratio - Whether to maintain frame aspect ratio or force into frame
    """

    def __init__(self, width, height, camera_link=-1, aspect_ratio=False, parent=None, deque_size=1, tracking=None):
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

        # Periodically set video frame to display
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.set_frame)
        self.timer.start(1)

        print('Started camera: {}'.format(self.camera_stream_link))

        # Camera Tracking for VISCA
        self.tracking = tracking

        # Facial Recognition & Object Tracking
        self.is_adding_face = False
        self.adding_to_name = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
        self.count = 0
        self.recognizer = None
        self.names = None
        self.resetFacialRecognition()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.id = 0

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

                            try:
                                if self.is_adding_face:
                                    frame = self.add_face(frame)
                                elif self.recognizer is not None:
                                    frame = self.recognize_face(frame)
                            except:
                                self.resetFacialRecognition()
                                print("resetting facial recognition")

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

                # print(self.count_for_reset)
                # break
                # if self.count_for_reset is None:
                #     self.count_for_reset = 0
                # elif self.count_for_reset < 2000:
                #     self.count_for_reset = self.count_for_reset + 1
                #     pass
                # else:
                #     self.kill_video()
                #     break

    def add_face(self, frame):
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            self.count = self.count + 1
            name = '../logic/facial_tracking/images/' + self.adding_to_name + '/' + str(self.count) + '.jpg'
            print("\n [INFO] Creating Images........." + name)
            cv2.imwrite(name, frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if self.count >= 50:  # Take 5000 face sample and stop video
            self.adding_to_name = None
            self.is_adding_face = False

            th = Thread(target=Trainer().train_face(False))
            th.daemon = True
            th.start()
            th.join()
            self.resetFacialRecognition()
            self.count = 0
            return frame
        else:
            return frame

    def resetFacialRecognition(self):
        if os.path.exists("../logic/facial_tracking/trainer/trainer.yml"):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                self.recognizer.read('../logic/facial_tracking/trainer/trainer.yml')
            except:
                self.resetFacialRecognition()

            # names related to ids: example ==> Steve: id=1 | try moving to trainer/labels.txt
            labels_file = open("../logic/facial_tracking/trainer/labels.txt", "r")
            self.names = labels_file.read().splitlines()
            labels_file.close()
        else:
            self.recognizer = None
            self.names = None

    def recognize_face(self, frame):
        # Define min window size to be recognized as a face
        minW = 0.1 * frame.shape[1]
        minH = 0.1 * frame.shape[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                   minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = self.recognizer.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                id = self.names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(frame, str(id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

        return frame

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
                                   QtGui.QImage.Format_RGB888).rgbSwapped()

                try:
                    self.video_frame.setPixmap(QtGui.QPixmap.fromImage(img))
                except:
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
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

    def get_video_frame(self):
        return self.video_frame

    def set_tracker(self, tracking):
        self.tracking = tracking

    def get_tracker(self):
        print(self.tracking)

    def kill_video(self):
        print("Killing Camera Object")

        with self.break_loop_lock:
            self.break_loop = True
        try:
            self.capture.release()
        except:
            pass
        cv2.destroyAllWindows()
        self.load_stream_thread = None
        self.capture = None
        self.online = False
        # self.video_frame.close()
        print("Camera Object Done")

    def config_add_face(self, name):
        self.adding_to_name = name
        self.is_adding_face = True
