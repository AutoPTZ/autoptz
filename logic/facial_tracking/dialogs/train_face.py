import os
import cv2
import face_recognition
import pickle
from imutils import paths

from shared.message_prompts import show_info_messagebox

class Trainer:
    @staticmethod
    def train_face(show_message_box):
        if show_message_box:
            show_info_messagebox("It will take a few seconds to minutes.\n Please Wait ...")
        print("\n [INFO] Training faces. It will take a few seconds to minutes. Please Wait ...")

        # Image path for face image database
        image_path = '../logic/facial_tracking/images/'
        trainer_loc = '../logic/facial_tracking/trainer/encodings.pickle'

        imagePaths = list(paths.list_images(image_path))
        knownEncodings = []
        knownNames = []

        # loop over the image paths
        for (i, imagePath) in enumerate(imagePaths):
            # extract the person name from the image path
            print(f"Processing {i + 1} of {len(imagePaths)}")
            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model='cnn')
            encodings = face_recognition.face_encodings(rgb, boxes)
            for encoding in encodings:
                knownEncodings.append(encoding)
                knownNames.append(name)
        # if show_message_box:
        #     show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(np.unique(ids))))
        # print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
        print("Saving encodings to encodings.pickle ...")
        data = {"encodings": knownEncodings, "names": knownNames}
        f = open(trainer_loc, "wb")
        f.write(pickle.dumps(data))
        f.close()
        print("Encodings have been saved successfully.")
        return


def main():
    Trainer.train_face(show_message_box=True)


if __name__ == '__main__':
    main()
