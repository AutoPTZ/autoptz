import math
import os
import pickle
from ctypes import c_char
from multiprocessing import Value
import cv2
import face_recognition
import numpy as np
import shared.constants as constants


class FacialRecognition:
    def __init__(self, queue, objectName):
        self.known_face_encodings = None
        self.queue = queue
        self.add_face_name = Value(c_char * 50)
        self.objectName = objectName
        self.check_encodings()

    def recognize(self, frame):
        if frame.shape[2] == 4:
            # Convert from BGR or BGRA to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Resize the frame to a smaller size
        small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)

        face_locations = face_recognition.face_locations(
            small_frame, number_of_times_to_upsample=1, model="hog")
        face_encodings = face_recognition.face_encodings(
            small_frame, face_locations, num_jitters=0, model="small")
        face_names = []
        confidence_list = []

        add_face_name = self.add_face_name.value.decode('utf-8')
        if add_face_name:
            result = self.add_face(face_encodings, add_face_name)
            if result:
                self.queue.put((face_locations, [add_face_name], [100]))
                return

            self.queue.put(([], [], []))
            return

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings['encodings'], face_encoding)
            name = "Unknown"
            confidence = ''

            face_distances = face_recognition.face_distance(
                self.known_face_encodings['encodings'], face_encoding)

            # Check if face_distances is empty
            if not face_distances.size:
                continue

            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_encodings['names'][best_match_index]
                confidence = self.face_confidence(
                    face_distances[best_match_index])
            face_names.append(name)
            confidence_list.append(confidence)
        face_details = (face_locations, face_names, confidence_list)
        self.queue.put(face_details)

    def set_add_face_name(self, name):
        self.add_face_name.value = name.encode('utf-8')

    def add_face(self, face_encodings, add_face_name):
        # If a face was found in the frame, add it to the known faces
        if face_encodings:
            # Add the new face encoding to the known faces
            self.known_face_encodings['encodings'].append(face_encodings[0])
            self.known_face_encodings['names'].append(add_face_name)

            # Save the updated known faces back to the file
            with open(constants.ENCODINGS_PATH, "wb") as f:
                f.write(pickle.dumps(self.known_face_encodings))

            print(f"Added a new face for {add_face_name}")
            # Reset the add_face_name to stop adding the face
            self.set_add_face_name('')
            return True
        return False

    @staticmethod
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

    def check_encodings(self):
        self.known_face_encodings = {'encodings': [], 'names': []}
        if os.path.exists(constants.ENCODINGS_PATH):
            encodings = pickle.loads(
                open(constants.ENCODINGS_PATH, "rb").read())
            self.known_face_encodings = encodings
