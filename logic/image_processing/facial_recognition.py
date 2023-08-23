import os
import pickle
import face_recognition
import cv2
import numpy as np

import shared.constants as constants


class FacialRecognition:
    def __init__(self, shared_data, objectName, recognition_interval=5):
        self.known_face_encodings = None
        self.shared_data = shared_data
        self.objectName = objectName
        self.recognition_interval = recognition_interval
        self.frame_count = 0
        self.pose_estimator = None
        self.model = None
        self.check_encodings()
        print("Facial Recognition and Pose Estimation service is starting")

    def recognize_and_estimate_pose(self, frame):
        # Increment frame count
        self.frame_count += 1

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.shared_data[f'{self.objectName}_pose_landmarks'] = None
        self.shared_data[f'{self.objectName}_facial_recognition_results'] = [
        ], [], []

        # Periodically recognize face
        if self.frame_count % self.recognition_interval == 0:
            face_locations = face_recognition.face_locations(
                rgb_frame, number_of_times_to_upsample=2, model="hog")
            face_encodings = face_recognition.face_encodings(
                rgb_frame, face_locations, num_jitters=1, model="small")
            face_names = []
            confidence_list = []

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

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings['encodings'], face_encoding)
                name = "Unknown"
                confidence = ''

                face_distances = face_recognition.face_distance(
                    self.known_face_encodings['encodings'], face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_encodings['names'][best_match_index]
                    confidence = self.face_confidence(
                        face_distances[best_match_index])
                face_names.append(name)
                confidence_list.append(confidence)
            self.shared_data[
                f'{self.objectName}_facial_recognition_results'] = face_locations, face_names, confidence_list

        # Estimate Pose
        results = self.pose_estimator.process(rgb_frame)
        if results.pose_landmarks:
            self.shared_data[f'{self.objectName}_pose_landmarks'] = results.pose_landmarks
        else:
            self.shared_data[f'{self.objectName}_pose_landmarks'] = None


        # # Body Detection
        # self.shared_data[f'{self.objectName}_body_detection_results'] = self.body_detection(
        #     frame)

    def body_detection(self, frame):
        if frame.shape[2] != 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 0.007843, (300, 300), 127.5)
        # blob = cv2.dnn.blobFromImage(frame)
        self.model.setInput(blob)
        detections = self.model.forward()
        body_detection_results = []
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if idx == 15:  # Assuming 15 is the class ID for humans
                    box = detections[0, 0, i, 3:7] * np.array(
                        [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                    body_detection_results.append(box.astype("int"))

        return body_detection_results

    def set_add_face_name(self, name):
        self.shared_data['add_face_name'] = name

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

    @staticmethod
    def face_confidence(face_distance, face_match_threshold=0.6):
        threshold = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (threshold * 2.0)

        if face_distance > face_match_threshold:
            return str(round(linear_val * 100, 2)) + '%'
        else:
            value = (linear_val + ((1.0 - linear_val) *
                     (linear_val - 0.5) * 2) ** 0.2) * 100
            return str(round(value, 2)) + '%'

    def check_encodings(self):
        self.shared_data[f'{self.objectName}_pose_landmarks'] = None
        self.shared_data[f'{self.objectName}_facial_recognition_results'] = [
        ], [], []
        self.known_face_encodings = {'encodings': [], 'names': []}
        if os.path.exists(constants.ENCODINGS_PATH):
            encodings = pickle.loads(
                open(constants.ENCODINGS_PATH, "rb").read())
            self.known_face_encodings = encodings
