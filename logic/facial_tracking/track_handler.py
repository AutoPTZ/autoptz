import pickle
import cv2
import os
import face_recognition
import numpy as np

TARGET_WH = 320
THRESHOLD = 0.85
NMS_THRESHOLD = 0.3


class TrackHandler:
    def __init__(self, face_loc=None, face_enc=None, face_names=None, process_this_frame=True):
        self.face_locations = face_loc
        self.face_encodings = face_enc
        self.face_names = face_names
        self.face_rec = face_recognition.FaceRec()
        self.process_this_frame = process_this_frame
        if os.path.exists("../logic/facial_tracking/trainer/encodings.pickle"):
            self.data = pickle.loads(open("../logic/facial_tracking/trainer/encodings.pickle", "rb").read())

        # # YoloV4
        # self.net = cv2.dnn.readNetFromDarknet('../logic/facial_tracking/trainer/yolo.cfg', '../logic/facial_tracking/trainer/yolov4-obj_final.weights')
        # self.classes = []
        # with open("../logic/facial_tracking/trainer/coco.names", "r") as f:
        #     self.classes = [line.strip() for line in f.readlines()]
        # layers_names = self.net.getLayerNames()
        # self.output_layers = [layers_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        # self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # MobileSSD
        # self.classNames = {0: 'background',
        #                    1: 'aeroplane', 2: 'bicycle', 3: 'bird', 4: 'boat',
        #                    5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat', 9: 'chair',
        #                    10: 'cow', 11: 'diningtable', 12: 'dog', 13: 'horse',
        #                    14: 'motorbike', 15: 'person', 16: 'pottedplant',
        #                    17: 'sheep', 18: 'sofa', 19: 'train', 20: 'tvmonitor'}
        # self.net = cv2.dnn.readNetFromCaffe('../logic/facial_tracking/trainer/MobileNetSSD_deploy.prototxt',
        #                                     '../logic/facial_tracking/trainer/MobileNetSSD_deploy.caffemodel')

        # self.net.setPreferableBackend(cv2.dnn.DNN_TARGET_CUDA)
        # self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    def resetFacialRecognition(self):
        if os.path.exists("../logic/facial_tracking/trainer/encodings.pickle"):
            self.data = pickle.loads(open("../logic/facial_tracking/trainer/encodings.pickle", "rb").read())

    def get_box_center(self, box):
        cx, cy = int((box[1] + box[3]) / 2), int((box[0] + box[2]) / 2)
        return cx, cy

    def getClassLabel(self, class_id, classes):
        for key, value in classes.items():
            if class_id == key:
                return value

    def mobile_ssd_detector(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        (h, w) = small_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(small_frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), False)
        # Set to network the input blob
        self.net.setInput(blob)
        # Prediction of network
        detections = self.net.forward()

        for detection in detections[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .5:
                class_id = detection[1]
                class_label = self.getClassLabel(class_id, self.classNames)
                if class_label == 'person':
                    x = int(detection[3] * w)
                    y = int(detection[4] * h)
                    w = int(detection[5] * w)
                    h = int(detection[6] * h)
                    cv2.rectangle(frame, (x * 4, y * 4, w * 4, h * 4), (0, 255, 0), thickness=2)
                    cv2.putText(frame, class_label, (x * 4, y * 4 + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3,
                                cv2.LINE_AA)

        return frame

    def yolo_detector_slow(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        blob = cv2.dnn.blobFromImage(small_frame, scalefactor=0.00392, size=(320, 320), mean=(0, 0, 0), swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.output_layers)

        height, width, channels = small_frame.shape

        boxes = []
        confs = []
        class_ids = []
        for output in outputs:
            for detect in output:
                scores = detect[5:]
                class_id = np.argmax(scores)
                conf = scores[class_id]
                if conf > 0.3:
                    center_x = int(detect[0] * width)
                    center_y = int(detect[1] * height)
                    w = int(detect[2] * width)
                    h = int(detect[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confs.append(float(conf))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i]
                cv2.rectangle(frame, (x * 4, y * 4), (x * 4 + w * 4, y * 4 + h * 4), color, 2)
                cv2.putText(frame, label, (x * 4, y * 4 - 5), font, 1, color, 1)
        return frame

    def yolo_detector_faster(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        blob = cv2.dnn.blobFromImage(small_frame, 1 / 255.0, (TARGET_WH, TARGET_WH),
                                     [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(output_names)

        hT, wT, cT = small_frame.shape
        bbox = []
        class_ids = []
        confs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]  # remove first five elements.
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > THRESHOLD:
                    box = detection[0:4] * np.array([wT, hT, wT, hT])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    bbox.append([x, y, int(width), int(height)])
                    class_ids.append(class_id)
                    confs.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, confs, THRESHOLD, NMS_THRESHOLD)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(bbox)):
            if i in indices:
                x, y, w, h = bbox[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[i]
                cv2.rectangle(frame, (x * 4, y * 4), (x*4 + w*4, y*4 + h*4), color, 2)
                cv2.putText(frame, label, (x*4, y*4 - 5), font, 1, color, 1)
        return frame

    def recognize_face(self, frame):

        if self.process_this_frame:
            # Resize frame of video to 1/2 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces and face encodings in the current frame of video
            self.face_locations = self.face_rec.face_locations(rgb_small_frame)
            self.face_encodings = self.face_rec.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []
            for face_encoding in self.face_encodings:
                # See if the face is a match for the known face(s)
                matches = self.face_rec.compare_faces(self.data['encodings'], face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = self.face_rec.face_distance(self.data['encodings'], face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.data['names'][best_match_index]
                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        return self.face_locations, self.face_names

    def show_yolo_bboxes(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (TARGET_WH, TARGET_WH),
                                     [0, 0, 0], 1, crop=False)
        self.net.setInput(blob)

        layer_names = self.net.getLayerNames()
        output_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        outputs = self.net.forward(output_names)

        hT, wT, cT = frame.shape
        bbox = []
        class_ids = []
        confs = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]  # remove first five elements.
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > THRESHOLD:
                    box = detection[0:4] * np.array([wT, hT, wT, hT])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    bbox.append([x, y, int(width), int(height)])
                    class_ids.append(class_id)
                    confs.append(float(confidence))
        indices = cv2.dnn.NMSBoxes(bbox, confs, THRESHOLD, NMS_THRESHOLD)

        for index in indices:
            index = index[0]  # the index has extra brackets so we have to remove them.
            class_name_index = class_ids[index]

            box = bbox[index]
            (x, y, w, h) = box[0:4]
            label = self.CLASSNAMES[class_name_index]
            conf = confs[index]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cx = int((x + x + w) / 2)
            cy = int((y + y + h) / 2)
            # constants.append("yolo_points", (cx, cy), cam_index)
            cv2.putText(frame, f'{label} {int(conf * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                        (255, 0, 0), 1)
        return frame
