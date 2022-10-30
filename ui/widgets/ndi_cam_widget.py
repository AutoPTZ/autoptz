import os.path
from collections import deque
from threading import Thread, Lock
import time
import NDIlib as ndi
import dlib
import numpy as np

import cv2
import imutils
from PyQt5 import QtCore, QtGui, QtWidgets

from logic.facial_tracking.train_face import Trainer


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
        self.break_loop_lock = Lock()
        self.break_loop = False
        self.load_stream_thread = None
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
        self.name_id = None
        self.enable_track_checked = False
        self.tracked_name = None
        self.track_started = None
        self.tracker = None
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None

        # ONVIF PTZ Control
        self.camera_control = None
        self.movementX = False

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
            with self.break_loop_lock:
                if self.break_loop:
                    break
                else:
                    try:
                        timer = cv2.getTickCount()
                        t, v, _, _ = ndi.recv_capture_v3(self.ndi_recv, 5000)
                        if self.online:
                            # Read next frame from stream and insert into deque
                            try:
                                if t == ndi.FRAME_TYPE_VIDEO:
                                    frame = np.copy(v.data)

                                    # Keep frame aspect ratio
                                    if self.maintain_aspect_ratio:
                                        frame = imutils.resize(frame, width=self.screen_width)
                                    # Force resize
                                    else:
                                        frame = cv2.resize(frame, (self.screen_width, self.screen_height))

                                    if self.is_adding_face:
                                        frame = self.add_face(frame)
                                    elif self.recognizer is not None:
                                        frame = self.recognize_face(frame)

                                    # try:
                                    #     if self.is_adding_face:
                                    #         frame = self.add_face(frame)
                                    #     elif self.recognizer is not None:
                                    #         frame = self.recognize_face(frame)
                                    # except:
                                    #     self.resetFacialRecognition()
                                    #     print("resetting facial recognition")

                                    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                                    frame = cv2.putText(frame, str(int(fps)), (75, 50), self.font, 0.7, (0, 0, 255), 2)

                                    # try:
                                    #     # Keep frame aspect ratio
                                    #     if self.maintain_aspect_ratio:
                                    #         frame = imutils.resize(frame, width=self.screen_width)
                                    #     # Force resize
                                    #     else:
                                    #         frame = cv2.resize(frame, (self.screen_width, self.screen_height))
                                    #
                                    #     if self.is_adding_face:
                                    #         frame = self.add_face(frame)
                                    #     elif self.recognizer is not None:
                                    #         frame = self.recognize_face(frame)
                                    # except:
                                    #     self.resetFacialRecognition()
                                    #     print("resetting facial recognition")
                                    #
                                    # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
                                    # frame = cv2.putText(frame, str(int(fps)), (75, 50), self.font, 0.7, (0, 0, 255),2)
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

    def add_face(self, frame):
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            self.count = self.count + 1
            name = '../logic/facial_tracking/images/' + self.adding_to_name + '/' + str(self.count) + '.jpg'
            print("\n [INFO] Creating Images........." + name)
            cv2.imwrite(name, frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if self.count >= 200:  # Take 5000 face sample and stop video
            self.adding_to_name = None
            self.is_adding_face = False

            # MacOS only allows UI things to show on the main thread.
            # Since this camera is on a separate thread,
            # we can't automatically train model here nor put it on its own thread
            # result = Trainer().train_face()
            # if result == "done":
            #
            # else:
            #     print(result)

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
                self.name_id = self.names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                self.name_id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            if self.name_id == self.tracked_name:
                self.track_x = x
                self.track_y = y
                self.track_w = w
                self.track_h = h
            cv2.putText(frame, str(self.name_id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

        if len(faces) == 0:
            self.name_id = "none"

        if self.enable_track_checked:
            frame = self.track_face(frame, self.track_x, self.track_y, self.track_w, self.track_h)

        return frame

    def track_face(self, frame, x, y, w, h):
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, "Tracking Enabled", (75, 75), self.font, 0.7, (0, 0, 255), 2)
        if not self.track_started:
            self.tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(x, y, x + w, y + h)
            self.tracker.start_track(rgbFrame, rect)
            self.track_started = True
            cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 255), 3, 1)
        if self.name_id == self.tracked_name:
            rect = dlib.rectangle(x, y, x + w, y + h)
            self.tracker.start_track(rgbFrame, rect)
            cv2.rectangle(frame, (int(x), int(y)), (int(w + x), int(h + y)), (255, 0, 255), 3, 1)
            cv2.putText(frame, "tracking", (int(x), int(h + 15)), self.font, 0.45, (0, 255, 0), 1)
        else:
            self.tracker.update(rgbFrame)
            pos = self.tracker.get_position()
            # unpack the position object
            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())
            cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (255, 0, 255), 3, 1)
            cv2.putText(frame, "tracking", (int(startX), int(endY + 15)), self.font, 0.45, (0, 255, 0), 1)

        if self.camera_control is not None:
            if x > 217 and x < 423:
                if self.movementX:
                    self.camera_control.stop_move()
                    self.movementX = False

            if not self.movementX:
                if x > 423:
                    self.camera_control.continuous_move(0.05, 0, 0)
                    self.movementX = True
                    print("Out of Best Bounds")
                elif x < 217:
                    self.camera_control.continuous_move(-0.05, 0, 0)
                    self.movementX = True
                    print("Out of Best Bounds")

        return frame

    def changeFace(self, name):
        if name == '':
            self.tracked_name = None
        else:
            self.tracked_name = name

    def checkFace(self):
        if self.tracked_name is None:
            return 'nothing'
        else:
            return self.tracked_name

    def spin(self, seconds):
        """Pause for set amount of seconds, replaces time.sleep so program doesnt stall"""

        time_end = time.time() + seconds
        while time.time() < time_end:
            QtWidgets.QApplication.processEvents()

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
                                   QtGui.QImage.Format_RGBX8888).rgbSwapped()

                try:
                    self.video_frame.setPixmap(QtGui.QPixmap.fromImage(img))
                except:
                    self.kill_video()

    def get_video_frame(self):
        return self.video_frame

    def config_add_face(self, name):
        self.adding_to_name = name
        self.is_adding_face = True

    def config_camera_control(self, control):
        self.camera_control = control

    def is_ptz_ready(self):
        if self.camera_control is None:
            return "not ready"
        else:
            return "ready"

    def config_enable_track(self):
        self.enable_track_checked = not self.enable_track_checked
        self.movementX = False

    def is_track_enabled(self):
        return self.enable_track_checked

    def kill_video(self):
        print("Killing Camera Object")

        with self.break_loop_lock:
            self.break_loop = True
        try:
            ndi.recv_destroy(self.ndi_recv)
            ndi.destroy()
        except:
            pass
        cv2.destroyAllWindows()
        self.load_stream_thread = None
        self.capture = None
        self.online = False
        print("Camera Object Done")
