import cv2
import os

ROOT_DIR = os.path.abspath(os.curdir)
IMAGE_PATH = ROOT_DIR + '/logic/facial_tracking/images/'
TRAINER_PATH = ROOT_DIR + "/logic/facial_tracking/trainer/"
ENCODINGS_PATH = ROOT_DIR + '/logic/facial_tracking/trainer/encodings.pickle'
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
FONT = cv2.FONT_HERSHEY_SIMPLEX
CAMERA_STYLESHEET = """
                    QLabel[active="false"]{
                        border: 2.5px solid slategray;
                        border-radius: 3px;}

                    QLabel::hover {
                        border: 2.5px solid crimson;
                        border-radius: 3px;}

                    QLabel[active="true"]{
                        border: 2.5px solid dodgerblue;
                        border-radius: 3px;}
                    """
CURRENT_ACTIVE_CAM_WIDGET = None