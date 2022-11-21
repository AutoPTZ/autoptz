import cv2
import shared.constants as constants
from threading import Thread
import os
import pickle
import math
import numpy as np
import time
from logic.facial_tracking.dialogs.train_face import TrainerDlg
import libraries.face_recognition as face_rec


class ImageProcessor:
    def __init__(self, stream_thread, width=None, height=None):
        super().__init__()
        self.stream = stream_thread
        self.width = width
        self.height = height
        self._run_flag = True

        # CameraWidget will access these three variables
        self.face_locations = None
        self.face_names = None
        self.confidence_list = None

        # Variables for Adding Faces, Recognition, and Tracking
        self.count = 0
        self.add_name = None
        self.tracking = None
        self.face_rec = face_rec.FaceRec()
        self.encoding_data = None

    def start(self):
        Thread(target=self.process, args=()).start()
        self.check_encodings()
        return self

    def process(self):
        while self._run_flag:
            frame = self.stream.cv_img
            if self.add_name:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.add_face(frame=frame, gray_frame=gray_frame)
            else:
                if self.encoding_data is not None:
                    self.recognize_face(frame)

    def add_face(self, frame, gray_frame):
        min_w = 0.1 * gray_frame.shape[1]
        min_h = 0.1 * gray_frame.shape[0]

        faces = constants.FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10,
                                                        minSize=(int(min_w), int(min_h)))
        time.sleep(0.07)  # add artificial timer sleep so users can see the boxes draw
        self.face_locations = []
        self.face_names = []
        self.confidence_list = []
        for x, y, w, h in faces:
            self.count = self.count + 1
            location = constants.IMAGE_PATH + self.add_name + '/' + str(self.count) + '.jpg'
            print("\n [INFO] Creating Images at " + location)
            cv2.imwrite(location, frame)
            self.face_names.append("Adding: " + self.add_name)
            self.face_locations = [(int(y / 2), int((x + w) / 2), int((y + h) / 2), int(x / 2))]
            self.confidence_list.append("100%")

        if self.count >= 10:  # Take 50 face sample and stop video
            self.add_name = None
            self.count = 0
            self.face_locations = None
            self.face_names = None
            # send signal for TrainingDlg

    @staticmethod
    def face_confidence(face_distance, face_match_threshold=0.6):
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
            return str(round(value, 2)) + '%'

    def recognize_face(self, frame):
        # Resize frame of video to 1/2 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        self.face_locations = self.face_rec.face_locations(rgb_small_frame)
        face_encodings = self.face_rec.face_encodings(rgb_small_frame, self.face_locations)

        self.face_names = []
        self.confidence_list = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = self.face_rec.compare_faces(self.encoding_data['encodings'], face_encoding)
            name = "Unknown"
            confidence = ''
            # Or instead, use the known face with the smallest distance to the new face
            face_distances = self.face_rec.face_distance(self.encoding_data['encodings'], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.encoding_data['names'][best_match_index]
                confidence = self.face_confidence(face_distances[best_match_index], 0.6)
            self.face_names.append(name)
            self.confidence_list.append(confidence)

    def check_encodings(self):
        self.encoding_data = None
        if os.path.exists(constants.ENCODINGS_PATH):
            self.encoding_data = pickle.loads(open(constants.ENCODINGS_PATH, "rb").read())

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
