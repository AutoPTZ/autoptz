from collections import OrderedDict
import pickle

CLASSESFILE = 'FaceRecognizer/files/class.names'
MODELCONFIG = 'FaceRecognizer/files/yolo.cfg'
MODELWEIGHTS = 'FaceRecognizer/files/yolo.weights'
ENCODINGS_PATH = 'FaceRecognizer/files/encodings.pickle'
FACE_DATASET_PATH = 'FaceRecognizer/files/face_dataset'
DETECTION_METHOD = 'cnn'
TARGET_WH = 320
THRESHOLD = 0.85
NMS_THRESHOLD = 0.3
DATA = pickle.loads(open(ENCODINGS_PATH, "rb").read())
with open(CLASSESFILE, 'rt') as f: CLASSNAMES = f.read().rstrip('\n').split('\n')


cams_in_use = []

yolo_points = OrderedDict()
recognizer_points = OrderedDict()
person_ids = OrderedDict()
missing_people = OrderedDict()

pt = OrderedDict()
# ct = CentroidTracker()
dicts = {"yolo_points": yolo_points,
         "recognizer_points": recognizer_points,
         "person_ids": person_ids}


def initialize():
    for i in cams_in_use:
        yolo_points[i] = []
        recognizer_points[i] = []
        missing_people[i] = OrderedDict()
        person_ids[i] = []
        pt[i] = PersonTracker()


# generalized function to add to dicts
def append(dict_name, to_add, cam_idx):
    dicts[dict_name][cam_idx].append(to_add)


def clear_ordered_dicts():
    for i in cams_in_use:
        yolo_points[i].clear()
        recognizer_points[i].clear()
        person_ids[i].clear()