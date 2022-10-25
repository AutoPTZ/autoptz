import os
import cv2
import numpy as np
from PIL import Image
from PyQt5.QtWidgets import QWidget, QInputDialog, QLineEdit, QLabel, QPushButton, QDialogButtonBox, QVBoxLayout, \
    QMessageBox

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")


def enterNameDialog():
    # window = QWidget().availableGeometry().center()
    # window.setGeometry(400, 400, 250, 250)
    # window.setWindowTitle("Register Face Process")
    text, pressed = QInputDialog.getText(QWidget(), "Register Face Process", "Enter Your Name: ",
                                         QLineEdit.Normal, "")
    if pressed:
        return text


def show_critical_messagebox():
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Critical)
    # setting message for Message Box
    msg.setText("Register Face Process")
    # setting Message box window title
    msg.setWindowTitle("Critical MessageBox")
    # declaring buttons on Message Box
    msg.setStandardButtons(QMessageBox.Ok)
    # start the app
    retval = msg.exec_()


def show_info_messagebox(string):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Information)
    # setting message for Message Box
    msg.setText(string)
    # setting Message box window title
    msg.setWindowTitle("Information")
    # declaring buttons on Message Box
    msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
    # start the app
    retval = msg.exec_()


def register_person():
    print("\n [INFO] Attempting To Draw Frame")
    video = cv2.VideoCapture(2)

    cv2.startWindowThread()
    print("\n [INFO] Frame Drawn")

    count = 0
    # Initialize individual sampling face count

    nameID = enterNameDialog()
    path = './images/' + nameID
    isExist = os.path.exists(path)

    while isExist:
        show_critical_messagebox()
        print("\n [INFO] Name Already Taken")
        # allow user to add more to the image list
        nameID = enterNameDialog()
    else:
        os.makedirs(path)

    show_info_messagebox("Initializing face capture. \nLook the camera and wait...")
    print("\n [INFO] Initializing face capture. Look the camera and wait ...")

    while True:
        ret, frame = video.read()
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        for x, y, w, h in faces:
            count = count + 1
            name = './images/' + nameID + '/' + str(count) + '.jpg'
            print("\n [INFO] Creating Images........." + name)
            cv2.imwrite(name, frame[y:y + h, x:x + w])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv2.imshow("Registering Face", frame)

        if cv2.waitKey(30) & 0xff == 27:
            break
        elif count >= 500:  # Take 5000 face sample and stop video
            break

    # Do a bit of cleanup
    # messagebox.showinfo("Register Face Process", "Exiting Facial Registration, Cleaning Up..", parent=root)
    show_info_messagebox("Exiting Facial Registration, Cleaning Up...")
    print("\n [INFO] Exiting Facial Registration, Cleaning Up...")
    video.release()
    cv2.destroyAllWindows()

    # Start Trainer for Faces
    train_face()
    # Start Basic Recognition Tracking
    recognize_face()
    return 0


def train_face():
    show_info_messagebox("It will take a few seconds to minutes.\n Please Wait ...")
    # messagebox.showinfo("Training Faces Process", "It will take a few seconds to minutes. Please Wait ...", parent=root)
    print("\n [INFO] Training faces. It will take a few seconds to minutes. Please Wait ...")

    # Path for face image database
    path = './images'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faceSamples = []
    ids = []

    print(os.listdir(path))
    labels_file = open("./trainer/labels.txt", "w")

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
    recognizer.save('./trainer/trainer.yml')  # recognizer.write() worked on Pi
    # Print the numer of faces trained and end program
    # messagebox.showinfo("Training Faces Process",
    #  "{0} faces trained. Opening Basic Recognition Software".format(len(np.unique(ids))),
    # parent=root)
    show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(np.unique(ids))))
    print("\n [INFO] {0} faces trained. Opening Basic Recognition Software".format(len(np.unique(ids))))
    return 0


def recognize_face():
    # messagebox.showinfo("Basic Recognition Software", "Opening Basic Recognition Software", parent=ROOT)
    print("\n [INFO] Opening Basic Recognition Software")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('./trainer/trainer.yml')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml");

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # names related to ids: example ==> Steve: id=1 | try moving to trainer/labels.txt
    labels_file = open("./trainer/labels.txt", "r")
    names = labels_file.read().splitlines()
    print(names)
    labels_file.close()

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(2)

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('Basic Recognition Software', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def main():
    register_person()
    # train_face()
    # recognize_face()


if __name__ == '__main__':
    main()
