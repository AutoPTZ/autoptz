import datetime
import pickle

import cv2
import face_recognition
import imutils
import numpy as np
from collections import OrderedDict

net = cv2.dnn.readNetFromDarknet('../logic/facial_tracking/trainer/yolo.cfg',
                                 '../logic/facial_tracking/trainer/yolov4-obj_final.weights')
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

data = pickle.loads(open("../logic/facial_tracking/trainer/encodings.pickle", "rb").read())

TARGET_WH = 320
THRESHOLD = 0.85
NMS_THRESHOLD = 0.3


class TrackHandler:

    def __init__(self):
        self.scale = 0.0
        self.drawing_frames = OrderedDict()

    def scale_box(self, box):
        return (int(x * self.scale) for x in box)

    def get_box_center(self, box):
        cx, cy = int((box[1] + box[3]) / 2), int((box[0] + box[2]) / 2)
        return cx, cy

    def get_recognized_face_names(self, rgb, data):
        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []
        for (index, encoding) in enumerate(encodings):
            name = ""
            matches = face_recognition.compare_faces(data["encodings"], encoding, tolerance=0.6)
            if True in matches:
                matched_indexes = [index for (index, match) in enumerate(matches) if match]
                name_counts = {}

                for i in matched_indexes:
                    name = data["names"][i]
                    name_counts[name] = name_counts.get(name) + 1 if name in name_counts.keys() else 1

                name = max(name_counts, key=name_counts.get)
            if name != "":
                names.append(name)
        return boxes, names

    def show_recognized_faces(self, frame):
        self.drawing_frames = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(rgb, width=750)
        self.scale = frame.shape[1] / float(rgb.shape[1])
        (boxes, names) = self.get_recognized_face_names(rgb, data)
        for (box, name) in zip(boxes, names):
            (top, right, bottom, left) = self.scale_box(box)
            cx, cy = self.get_box_center(box)
            y = top - 15 if top - 15 > 15 else top + 15
            # constants.append("recognizer_points", (cx, cy), cam_id)
            # constants.append("person_ids", self.get_id(name), cam_id)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)
            cv2.rectangle(frame, (left, y), (right, y + 15), (0, 0, 255), -1)
            cv2.putText(frame, name, (left, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def get_concatenated_frames(self):
        return np.concatenate([value for _, value in self.drawing_frames.items()], axis=1)
