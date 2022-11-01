import os
import cv2
from threading import Thread
import dlib

from logic.facial_tracking.dialogs.train_face import Trainer
from ui.homepage.move_visca_ptz import ViscaPTZ


class ImageProcessor(Thread):

    def __init__(self):
        Thread.__init__(self)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
        self.image_path = '../logic/facial_tracking/images/'
        self.trainer_path = '../logic/facial_tracking/trainer/trainer.json'

        # Facial Recognition & Object Tracking
        self.is_adding_face = False
        self.adding_to_name = None
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

        # VISCA/ONVIF PTZ Control
        self.ptz_ready = None
        self.camera_control = None

    def get_frame(self, frame):
        if self.is_adding_face:
            return self.add_face(frame)
        if self.recognizer is not None:
            frame = self.recognize_face(frame)
        if self.enable_track_checked and self.track_x is not None and self.track_y is not None and self.track_w is not None and self.track_h is not None:
            frame = self.track_face(frame, self.track_x, self.track_y, self.track_w, self.track_h)
        return frame

    def add_face(self, frame):
        faces = self.face_cascade.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            self.count = self.count + 1
            name = self.image_path + self.adding_to_name + '/' + str(self.count) + '.jpg'
            print("\n [INFO] Creating Images........." + name)
            cv2.imwrite(name, frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        if self.count >= 50:  # Take 50 face sample and stop video
            self.adding_to_name = None
            self.is_adding_face = False

            trainer_thread = Thread(target=Trainer().train_face(False))
            trainer_thread.daemon = True
            trainer_thread.start()
            trainer_thread.join()
            self.resetFacialRecognition()
            self.count = 0
            return frame
        else:
            return frame

    def recognize_face(self, frame):
        # Define min window size to be recognized as a face
        minW = 0.1 * frame.shape[1]
        minH = 0.1 * frame.shape[0]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5,
                                                   minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            try:
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
            except:
                self.resetFacialRecognition()
                self.spin(2)
                return frame
            cv2.putText(frame, str(self.name_id), (x + 5, y - 5), self.font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), self.font, 1, (255, 255, 0), 1)

        if len(faces) == 0:
            self.name_id = "none"

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
            if 217 < x < 423:
                self.camera_control.move_stop()
            if x > 423:
                self.camera_control.move_right_track()
                print("Out of Best Bounds")
            elif x < 217:
                self.camera_control.move_left_track()
                print("Out of Best Bounds")
        return frame

    def resetFacialRecognition(self):
        self.names = []
        self.recognizer = None
        if os.path.exists(self.trainer_path):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(self.trainer_path)
            self.names = []
            for folder in os.listdir(self.image_path):
                self.names.append(folder)

    def config_add_face(self, name):
        self.adding_to_name = name
        self.is_adding_face = True

    def set_face(self, name):
        self.tracked_name = name

    def get_face(self):
        return self.tracked_name

    def set_ptz_ready(self, text):
        self.ptz_ready = text

    def get_ptz_ready(self):
        return self.ptz_ready

    def config_enable_track(self):
        self.enable_track_checked = not self.enable_track_checked

    def is_track_enabled(self):
        return self.enable_track_checked

    def set_ptz_tracker(self, control, isVISCA):
        if control is not None and isVISCA:
            self.camera_control = ViscaPTZ(device_id=control)
        else:
            self.camera_control = None