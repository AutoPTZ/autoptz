import os

import cv2
from logic.facial_tracking.utils import train_faces

from ui.shared.message_prompts import show_info_messagebox

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")


class Trainer:
    @staticmethod
    def train_face(show_message_box):
        if show_message_box:
            show_info_messagebox("It will take a few seconds to minutes.\n Please Wait ...")
        print("\n [INFO] Training faces. It will take a few seconds to minutes. Please Wait ...")

        # Image path for face image database
        image_path = '../logic/facial_tracking/images/'

        train_faces()

        if show_message_box:
            show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(os.listdir(image_path))))
        print("\n [INFO] {0} faces trained.".format(len(os.listdir(image_path))))
        return


def main():
    Trainer.train_face(show_message_box=True)


if __name__ == '__main__':
    main()
