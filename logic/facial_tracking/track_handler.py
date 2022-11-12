import datetime
import pickle

import cv2
import os
import face_recognition
import imutils
import numpy as np
from collections import OrderedDict

# net = cv2.dnn.readNetFromDarknet('../logic/facial_tracking/trainer/yolo.cfg',
#                                  '../logic/facial_tracking/trainer/yolov4-obj_final.weights')
# net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

TARGET_WH = 320
THRESHOLD = 0.85
NMS_THRESHOLD = 0.3


class TrackHandler:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True
        if os.path.exists("../logic/facial_tracking/trainer/encodings.pickle"):
            self.data = pickle.loads(open("../logic/facial_tracking/trainer/encodings.pickle", "rb").read())

    def get_box_center(self, box):
        cx, cy = int((box[1] + box[3]) / 2), int((box[0] + box[2]) / 2)
        return cx, cy

    def recognize_face(self, frame):

        if self.process_this_frame:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.data['encodings'], face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.data['encodings'], face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.data['names'][best_match_index]
                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        return frame
