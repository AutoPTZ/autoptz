import os

import cv2
import numpy as np
from PIL import Image

from ui.shared.message_prompts import show_info_messagebox

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")


class Trainer:

    @staticmethod
    def train_face():
        #show_info_messagebox("It will take a few seconds to minutes.\n Please Wait ...")
        print("\n [INFO] Training faces. It will take a few seconds to minutes. Please Wait ...")

        # Path for face image database
        path = '../logic/facial_tracking/images/'

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        faceSamples = []
        ids = []

        print(os.listdir(path))
        labels_file = open("../logic/facial_tracking/trainer/labels.txt", "w")

        for folder in os.listdir(path):
            current_folder = path + '/' + folder
            print("\n [INFO] Looking at " + current_folder + " now")
            labels_file.write(folder + "\n")
            for image in os.listdir(current_folder):
                PIL_img = Image.open(current_folder + '/' + image).convert('L')  # convert it to grayscale
                img_numpy = np.array(PIL_img, 'uint8')
                id = os.listdir(path).index(folder)
                faces = face_cascade.detectMultiScale(img_numpy)

                for (x, y, w, h) in faces:
                    faceSamples.append(img_numpy[y:y + h, x:x + w])
                    ids.append(id)
        labels_file.close()

        # Send to trainer
        recognizer.train(faceSamples, np.array(ids))
        # Save the model into trainer/trainer.yml
        recognizer.save('../logic/facial_tracking/trainer/trainer.yml')  # recognizer.write() worked on Pi
        # Print the numer of faces trained and end program
        # messagebox.showinfo("Training Faces Process",
        #  "{0} faces trained. Opening Basic Recognition Software".format(len(np.unique(ids))),
        # parent=root)
        #show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(np.unique(ids))))
        print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
        return "done"
# def run_trainer(recognizer):
#     th = threading.Thread(target=train_face)
#     th.start()
#
#     th.join()
#     recognizer =s
