import cv2
import os

ROOT_DIR = os.path.abspath(os.curdir)
IMAGE_PATH = ROOT_DIR + '/logic/facial_tracking/images/'
TRAINER_PATH = ROOT_DIR + "/logic/facial_tracking/trainer/"
ENCODINGS_PATH = ROOT_DIR + '/logic/facial_tracking/trainer/encodings.pickle'
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
FONT = cv2.FONT_HERSHEY_SIMPLEX
