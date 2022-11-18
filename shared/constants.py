import cv2

IMAGE_PATH = '../logic/facial_tracking/images/'
TRAINER_PATH = "../logic/facial_tracking/trainer/"
ENCODINGS_PATH = '../logic/facial_tracking/trainer/encodings.pickle'
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")
