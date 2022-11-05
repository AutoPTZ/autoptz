import dlib
from glob import glob
import cv2
import numpy as np
import os
import pickle

# load the face detector, landmark predictor, and face recognition model
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("../logic/facial_tracking/models/shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1(
    "../logic/facial_tracking/models/dlib_face_recognition_resnet_model_v1.dat")

# change this to include other image formats you want to support (e.g. .webp)
VALID_EXTENSIONS = ['.png', '.jpg', '.jpeg']


def train_faces():
    root_dir = "../logic/facial_tracking/images/"
    class_names = os.listdir(root_dir)

    # get the paths to the images
    image_paths = get_image_paths(root_dir, class_names)
    # initialize a dictionary to store the name of each person and the corresponding encodings
    name_encondings_dict = {}

    # initialize the number of images processed
    nb_current_image = 1
    # now we can loop over the image paths, locate the faces, and encode them
    for image_path in image_paths:
        print(f"Image processed {nb_current_image}/{len(image_paths)}")
        # load the image
        image = cv2.imread(image_path)
        # get the face embeddings
        encodings = face_encodings(image)
        # get the name from the image path
        name = image_path.split(os.path.sep)[-2]
        # get the encodings for the current name
        e = name_encondings_dict.get(name, [])
        # update the list of encodings for the current name
        e.extend(encodings)
        # update the list of encodings for the current name
        name_encondings_dict[name] = e
        nb_current_image += 1

    # save the name encodings dictionary to disk
    with open("../logic/facial_tracking/encodings.pickle", "wb") as f:
        pickle.dump(name_encondings_dict, f)


def get_image_paths(root_dir, class_names):
    """ grab the paths to the images in our dataset"""
    image_paths = []

    # loop over the class names
    for class_name in class_names:
        # grab the paths to the files in the current class directory
        class_dir = os.path.sep.join([root_dir, class_name])
        class_file_paths = glob(os.path.sep.join([class_dir, '*.*']))

        # loop over the file paths in the current class directory
        for file_path in class_file_paths:
            # extract the file extension of the current file
            ext = os.path.splitext(file_path)[1]

            # if the file extension is not in the valid extensions list, ignore the file
            if ext.lower() not in VALID_EXTENSIONS:
                print("Skipping file: {}".format(file_path))
                continue

            # add the path to the current image to the list of image paths
            image_paths.append(file_path)

    return image_paths


def face_rects(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = face_detector(gray, 1)
    # return the bounding boxes
    return rects


def face_landmarks(image):
    return [shape_predictor(image, face_rect) for face_rect in face_rects(image)]


def face_encodings(image):
    # compute the facial embeddings for each face 
    # in the input image. the compute_face_descriptor 
    # function returns a 128-d vector that describes the face in an image
    return [np.array(face_encoder.compute_face_descriptor(image, face_landmark))
            for face_landmark in face_landmarks(image)]


def nb_of_matches(known_encodings, unknown_encoding):
    # compute the euclidean distance between the current face encoding 
    # and all the face encodings in the database
    distances = np.linalg.norm(known_encodings - unknown_encoding, axis=1)
    # keep only the distances that are less than the threshold
    small_distances = distances <= 0.6
    # return the number of matches
    return sum(small_distances)
