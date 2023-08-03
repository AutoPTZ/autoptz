import os
import pickle
import math
import numpy as np
import face_recognition
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
    def __init__(self, shared_data, objectName):
        self.known_face_encodings = None
        self.shared_data = shared_data
        self.objectName = objectName
        self.check_encodings()
        print("Facial Recognition starting")

    def set_add_face_name(self, name):
        self.shared_data['add_face_name'] = name

    def check_encodings(self):
        """
        Refresh encodings_data to use the latest models data. If there is any.
        """
        # Always reset the known_face_encodings first
        self.known_face_encodings = {'encodings': [], 'names': []}

        # Then load the encodings if the file exists
        if os.path.exists(constants.ENCODINGS_PATH):
            print("loading encoded model")
            encodings = pickle.loads(
                open(constants.ENCODINGS_PATH, "rb").read())
            self.known_face_encodings = encodings

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
            self.shared_data['add_face_name'] = None
            return True
        return False

    def recognize(self, frame):
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(
            rgb_frame, number_of_times_to_upsample=0, model="hog")
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations, num_jitters=1, model="small")

        # If no faces were found in the frame, return empty results
        if not face_encodings:
            self.shared_data[f'{self.objectName}_facial_recognition_results'] = [
            ], [], []
            return
        add_face_name = self.shared_data.get('add_face_name')
        if add_face_name is not None:
            result = self.add_face(face_encodings, add_face_name)
            if result:
                self.shared_data['add_face_name'] = None
                self.shared_data[f'{self.objectName}_facial_recognition_results'] = face_locations, [
                    add_face_name], [100]
                return

            self.shared_data[f'{self.objectName}_facial_recognition_results'] = [
            ], [], []
            return

        if self.known_face_encodings == {'encodings': [], 'names': []}:
            self.shared_data[f'{self.objectName}_facial_recognition_results'] = [
            ], [], []
            return

        face_names = []
        confidence_list = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                self.known_face_encodings['encodings'], face_encoding)
            name = "Unknown"
            confidence = ''

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                self.known_face_encodings['encodings'], face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_encodings['names'][best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
                face_names.append(name)
                confidence_list.append(confidence)

        self.shared_data[f'{self.objectName}_facial_recognition_results'] = face_locations, face_names, confidence_list
