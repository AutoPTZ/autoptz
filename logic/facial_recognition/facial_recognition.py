import math
import os
import pickle
import numpy as np
from libraries.face_recognition import FaceRec
import cv2
import shared.constants as constants


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


class FacialRecognition:
    def __init__(self):
        self.face_recognition = FaceRec()
        self.check_encodings()

    def check_encodings(self):
        """
        Refresh encodings_data to use the latest models data. If there is any.
        """
        self.known_face_encodings = None
        if os.path.exists(constants.ENCODINGS_PATH):
            self.known_face_encodings = pickle.loads(
                open(constants.ENCODINGS_PATH, "rb").read())

    def recognize(self, frame):

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = self.face_recognition.face_locations(rgb_frame)
        face_encodings = self.face_recognition.face_encodings(
            rgb_frame, face_locations)

        face_names = []
        confidence_list = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = self.face_recognition.compare_faces(
                self.known_face_encodings['encodings'], face_encoding)
            name = "Unknown"
            confidence = ''

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = self.face_recognition.face_distance(
                self.known_face_encodings['encodings'], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_encodings['names'][best_match_index]
                confidence = face_confidence(
                    face_distances[best_match_index])
            face_names.append(name)
            confidence_list.append(confidence)

        return face_locations, face_names, confidence_list
