import os
import cv2
import face_recognition
import pickle
from imutils import paths

from shared.message_prompts import show_info_messagebox

class Trainer:
    face_recognition = face_recognition.FaceRec()
    @staticmethod
    def train_face(show_message_box):
        if show_message_box:
            show_info_messagebox("It will take a few seconds to minutes.\n Please Wait ...")
        print("\n [INFO] Training faces. It will take a few seconds to minutes. Please Wait ...")

        # Image path for face image database
        image_path = '../logic/facial_tracking/images/'
        encodings_path = '../logic/facial_tracking/trainer/encodings.pickle'

        imagePaths = list(paths.list_images(image_path))
        knownEncodings = []
        knownNames = []

        if os.listdir(image_path):
            # loop over the image paths
            for (i, imagePath) in enumerate(imagePaths):
                # extract the person name from the image path
                print(f"Processing {i + 1} of {len(imagePaths)}")
                name = imagePath.split(os.path.sep)[-2]
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = Trainer.face_recognition.face_locations(rgb, model='cnn')
                encodings = Trainer.face_recognition.face_encodings(rgb, boxes)
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
            print("Saving encodings to encodings.pickle ...")
            data = {"encodings": knownEncodings, "names": knownNames}
            f = open(encodings_path, "wb")
            f.write(pickle.dumps(data))
            f.close()
            print("Encodings have been saved successfully.")
        else:
            print("No images to train.")
            if os.path.exists(encodings_path):
                os.remove(encodings_path)
        if show_message_box:
            show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(knownNames)))
        print("\n [INFO] {0} faces trained.".format(len(knownNames)))
        return


def main():
    Trainer.train_face(show_message_box=True)


if __name__ == '__main__':
    main()
