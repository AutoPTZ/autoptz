import cv2
from PySide6.QtCore import QThread, Signal
import shared.constants as constants
import os
import pickle
import math
import numpy as np
import time
from libraries.face_recognition import FaceRec


def face_confidence(face_distance, face_match_threshold=0.6):
    """
    Confidence calculation for Facial Recognition
    :param face_distance:
    :param face_match_threshold:
    :return:
    """
    threshold = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (threshold * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


class ImageProcessor(QThread):
    """
    Threaded ImageProcessor for CameraWidget.
    Used for added faces to database and facial recognition.
    """
    retrain_model_signal = Signal()
    stream = None
    _run_flag = None
    lock = None
    face_locations = None
    face_names = None
    confidence_list = None
    count = 0
    add_name = None
    face_rec = None
    encoding_data = None

    def __init__(self, stream_thread, lock):
        super().__init__()
        self.stream = stream_thread
        self._run_flag = True
        self.lock = lock
        self.daemon = True

        # CameraWidget will access these three variables for Facial Recognition
        self.face_locations = []
        self.face_names = []
        self.confidence_list = []

        # CameraWidget will access this variable for Body Detection
        self.body_locations = []

        # Load the model
        self.model = cv2.dnn.readNetFromCaffe(
            constants.PROTOTXT_PATH, constants.CAFFEMODEL_PATH)

        # Variables for Adding Faces, Recognition, and Tracking
        self.count = 0
        self.add_name = None
        self.face_rec = FaceRec()
        self.encoding_data = None
        self.check_encodings()
        self.skip_frame = True

    def run(self):
        """
        Runs continuously on CameraWidget.start() to provide the latest face boxes for
        CameraWidget to drawn until _run_flag is False.
        """
        while self._run_flag:
            self.lock.acquire(blocking=True)
            frame = self.stream.cv_img
            if frame is not None:
                if self.add_name is not None:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    self.add_face(frame=frame, gray_frame=gray_frame)
                elif self.encoding_data is not None:
                    try:
                        self.recognize_face(frame=frame)
                    except Exception as e:
                        print(e)
                # else:
                #     self.stop()
            if self.lock.locked():
                self.lock.release()

    def add_face(self, frame, gray_frame):
        """
        If there is a face to add, then use OpenCV Cascades to save images to database and send for training.
        :param frame: Is used for saving the complete original image.
        :param gray_frame: Is used for OpenCV face detection.
        """
        min_w = 0.1 * gray_frame.shape[1]
        min_h = 0.1 * gray_frame.shape[0]

        faces = constants.FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10,
                                                        minSize=(int(min_w), int(min_h)))
        self.face_locations = []
        self.face_names = []
        self.confidence_list = []
        # add artificial timer sleep so users can see the boxes draw
        time.sleep(0.07)
        for x, y, w, h in faces:
            self.count += 1
            location = constants.IMAGE_PATH + \
                self.add_name + '/' + str(self.count) + '.jpg'
            print("\n [INFO] Creating Images at " + location)
            cv2.imwrite(location, frame)
            self.face_names.append("Adding: " + self.add_name)
            self.face_locations = [
                (int(y), int((x + w)), int((y + h)), int(x))]
            self.confidence_list.append("100%")

        if self.count >= 10:  # Take 50 face sample and stop video
            self.add_name = None
            self.count = 0
            self.face_locations = None
            self.face_names = None
            self.retrain_model_signal.emit()

    def body_detection(self, frame):
        if frame is not None:
            if frame.shape[2] != 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            blob = cv2.dnn.blobFromImage(cv2.resize(
                frame, (300, 300)), 0.007843, (300, 300), 127.5)
            self.model.setInput(blob)
            detections = self.model.forward()
            self.body_locations = []
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.2:
                    idx = int(detections[0, 0, i, 1])
                    if idx == 15:  # Assuming 15 is the class ID for humans
                        box = detections[0, 0, i, 3:7] * np.array(
                            [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                        self.body_locations.append(box.astype("int"))

    def recognize_face(self, frame):
        """
        Runs grabbed frame through Facial Recognition Library and sets Face Locations, Names, and Confidences in a list.
        :param frame:
        """
        if frame is not None and self.skip_frame:
            # Resize frame of video to 1/2 size for faster face recognition processing
            # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Find all the faces and face encodings in the current frame of video
            self.face_locations = self.face_rec.face_locations(
                rgb_small_frame, number_of_times_to_upsample=0)
            face_encodings = self.face_rec.face_encodings(
                rgb_small_frame, self.face_locations)

            self.face_names = []
            self.confidence_list = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = self.face_rec.compare_faces(
                    self.encoding_data['encodings'], face_encoding)
                name = "Unknown"
                confidence = ''
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = self.face_rec.face_distance(
                    self.encoding_data['encodings'], face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.encoding_data['names'][best_match_index]
                    confidence = face_confidence(
                        face_distances[best_match_index])
                self.face_names.append(name)
                self.confidence_list.append(confidence)
            self.body_detection(frame)
        self.skip_frame = not self.skip_frame

    def check_encodings(self):
        """
        Refresh encodings_data to use the latest models data. If there is any.
        """
        self.encoding_data = None
        if os.path.exists(constants.ENCODINGS_PATH):
            self.encoding_data = pickle.loads(
                open(constants.ENCODINGS_PATH, "rb").read())

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.lock.release()
        self.wait()
        self.deleteLater()
