import multiprocessing
from multiprocessing import Process, Pool

import cv2
from PySide6.QtCore import QThread

import shared.constants as constants
from threading import Thread
import os
import pickle
import math
import numpy as np
import time
from libraries.face_recognition import FaceRec
from logic.facial_tracking.dialogs.train_face import TrainerDlg
from multiprocessing import Process


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
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


# def recognize_face(frame):
#     tic = timeit.default_timer()
#     encoding_data = pickle.loads(open(constants.ENCODINGS_PATH, "rb").read())
#     face_rec = FaceRec()
#     face_locations = []
#     face_names = []
#     confidence_list = []
#     if frame is not None:
#         # Resize frame of video to 1/2 size for faster face recognition processing
#         small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#
#         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#         rgb_small_frame = small_frame[:, :, ::-1]
#
#         # Find all the faces and face encodings in the current frame of video
#         face_locations = face_rec.face_locations(rgb_small_frame, number_of_times_to_upsample=0, model="cnn")
#         face_encodings = face_rec.face_encodings(rgb_small_frame, face_locations)
#         for face_encoding in face_encodings:
#             # See if the face is a match for the known face(s)
#             matches = face_rec.compare_faces(encoding_data['encodings'], face_encoding)
#             name = "Unknown"
#             confidence = ''
#             # Or instead, use the known face with the smallest distance to the new face
#             face_distances = face_rec.face_distance(encoding_data['encodings'], face_encoding)
#             best_match_index = np.argmin(face_distances)
#             if matches[best_match_index]:
#                 name = encoding_data['names'][best_match_index]
#                 confidence = face_confidence(face_distances[best_match_index])
#             face_names.append(name)
#             confidence_list.append(confidence)
#     toc = timeit.default_timer()
#     print(f'Done in {toc - tic}')
#     print(face_locations, face_names, confidence_list)
#     return face_locations, face_names, confidence_list


class ImageProcessor(QThread):
    """
    Threaded ImageProcessor for CameraWidget.
    Used for added faces to database and facial recognition for now.
    *** NEED TO ADD FACIAL TRACKING ***
    """

    def __init__(self, stream_thread):
        super().__init__()
        self.stream = stream_thread
        self._run_flag = True

        # CameraWidget will access these four variables for Facial Recognition (3) and Tracking (1)
        self.face_locations = None
        self.face_names = None
        self.confidence_list = None
        self.tracked_location = None

        # Variables for Adding Faces, Recognition, and Tracking
        self.count = 0
        self.add_name = None
        self.face_rec = FaceRec()
        self.encoding_data = None
        self.check_encodings()

    def run(self):
        """
        Runs continuously on CameraWidget.start() to provide the latest face boxes for CameraWidget to drawn until _run_flag is False.
        """
        while self._run_flag:
            frame = self.stream.cv_img
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self.add_name:
                self.add_face(frame=frame, gray_frame=gray_frame)
            elif self.encoding_data is not None:
                try:
                    self.recognize_face(frame)
                except Exception as e:
                    print(e)
                # p = Pool(processes=6)
                # data = p.map(recognize_face, [frame])
                #
                # for loc, name, conf in data:
                #     self.face_locations = loc
                #     self.face_names = name
                #     self.confidence_list = conf

                # recognition = Process(target=recognize_face, args=(frame,))
                # recognition.start()
                # recognition.join()
            else:  # Free up threads and fixes Window's performance issue with useless thread
                self.stop()

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
        time.sleep(0.07)  # add artificial timer sleep so users can see the boxes draw
        self.face_locations = []
        self.face_names = []
        self.confidence_list = []
        for x, y, w, h in faces:
            self.count += 1
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

    def recognize_face(self, frame):
        """
        Runs grabbed frame through Facial Recognition Library and sets Face Locations, Names, and Confidences in a list.
        :param frame:
        """
        if frame is not None:
            # Resize frame of video to 1/2 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            self.face_locations = self.face_rec.face_locations(rgb_small_frame, number_of_times_to_upsample=0)
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
                    confidence = face_confidence(face_distances[best_match_index], 0.6)
                self.face_names.append(name)
                self.confidence_list.append(confidence)

    def check_encodings(self):
        """
        Refresh encodings_data to use the latest trainer data. If there is any.
        """
        self.encoding_data = None
        if os.path.exists(constants.ENCODINGS_PATH):
            self.encoding_data = pickle.loads(open(constants.ENCODINGS_PATH, "rb").read())

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
