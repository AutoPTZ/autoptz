import os

import cv2
import numpy as np
from PIL import Image

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
        labels_loc = '../logic/facial_tracking/trainer/labels.txt'
        trainer_loc = '../logic/facial_tracking/trainer/trainer.json'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faceSamples = []
        ids = []

        labels_file = open(labels_loc, "w")

        for folder in os.listdir(image_path):
            current_folder = image_path + folder
            print("\n [INFO] Looking at " + current_folder + " now")
            labels_file.write(folder + "\n")
            for image in os.listdir(current_folder):
                PIL_img = Image.open(current_folder + '/' + image).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')
                id = os.listdir(image_path).index(folder)
                faces = face_cascade.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
        labels_file.close()

        try:
            # Send to trainer
            recognizer.train(faceSamples, np.array(ids))
            # Save the model into trainer/trainer.json
            if os.path.exists(trainer_loc):
                os.remove(trainer_loc)
            recognizer.write(trainer_loc)
        except:
            os.remove(trainer_loc)
        if show_message_box:
            show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(np.unique(ids))))
        print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
        return


def main():
    Trainer.train_face(show_message_box=True)


if __name__ == '__main__':
    main()
