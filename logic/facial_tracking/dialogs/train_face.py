import os
import cv2
from libraries import face_recognition
import pickle
from imutils import paths
import shared.constants as constants
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog


class TrainerUI(object):
    def __init__(self):
        self.training_progress_bar_title = None
        self.verticalLayout = None
        self.window = None
        self.progress_bar_line = None
        self.trainer_thread = None
        self.horizontalLayout = None
        self.count = 0

    def setMaximumVal(self, value):
        self.progress_bar_line.setMaximum(value)

    def setCurrentVal(self, value):
        self.progress_bar_line.setValue(value)
        if value == self.progress_bar_line.maximum():
            self.window.close()

    def setCurrentTitle(self, text):
        self.training_progress_bar_title.setText(text + " Images")

    def setupUi(self, training_progress_bar):
        self.window = training_progress_bar
        training_progress_bar.setObjectName("training_progress_bar")
        training_progress_bar.resize(325, 100)
        self.verticalLayout = QtWidgets.QVBoxLayout(training_progress_bar)
        self.verticalLayout.setObjectName("verticalLayout")
        self.training_progress_bar_title = QtWidgets.QLabel(training_progress_bar)
        self.verticalLayout.addWidget(self.training_progress_bar_title)

        self.trainer_thread = TrainerThread()
        self.trainer_thread.MAX_VALUE_SIGNAL.connect(self.setMaximumVal)
        self.trainer_thread.CURRENT_VALUE_SIGNAL.connect(self.setCurrentVal)
        self.trainer_thread.CURRENT_TEXT_SIGNAL.connect(self.setCurrentTitle)

        self.progress_bar_line = QtWidgets.QProgressBar(training_progress_bar)
        self.progress_bar_line.setObjectName("progress_bar_line")
        self.progress_bar_line.setStyleSheet(
            '''
            QProgressBar {
                text-align: center;
                border: 2px solid grey;
                border-radius: 5px;
            }

            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 20px;
            }
            '''
        )
        self.verticalLayout.addWidget(self.progress_bar_line)

        self.translate_ui(training_progress_bar)
        QtCore.QMetaObject.connectSlotsByName(training_progress_bar)
        self.trainer_thread.start()

    @staticmethod
    def translate_ui(training_progress_bar):
        _translate = QtCore.QCoreApplication.translate
        training_progress_bar.setWindowTitle(_translate("training_progress_bar", "Training Model"))


class TrainerThread(QtCore.QThread):
    MAX_VALUE_SIGNAL = QtCore.Signal(int)
    CURRENT_VALUE_SIGNAL = QtCore.Signal(int)
    CURRENT_TEXT_SIGNAL = QtCore.Signal(str)
    DONE_SIGNAL = QtCore.Signal(bool)
    face_recognition = face_recognition.FaceRec()

    def __init__(self):
        super(TrainerThread, self).__init__()

    def __del__(self):
        self.wait()

    def run(self):
        print("\n [INFO] Training faces. It will take a few minutes. Please Wait ...")
        # Image path for face image database
        image_paths = list(paths.list_images(constants.IMAGE_PATH))
        known_encodings = []
        known_names = []
        self.MAX_VALUE_SIGNAL.emit(len(image_paths))
        self.CURRENT_VALUE_SIGNAL.emit(0)
        if os.listdir(constants.IMAGE_PATH):
            # loop over the image paths
            for (i, imagePath) in enumerate(image_paths):
                # extract the person name from the image path
                print(f"Processing {i + 1} of {len(image_paths)}")
                self.CURRENT_TEXT_SIGNAL.emit(f"Processing {i + 1} of {len(image_paths)}")
                self.CURRENT_VALUE_SIGNAL.emit(i + 1)
                name = os.path.basename(os.path.dirname(imagePath))
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = self.face_recognition.face_locations(rgb)
                encodings = self.face_recognition.face_encodings(rgb, boxes)
                for encoding in encodings:
                    known_encodings.append(encoding)
                    known_names.append(name)
            print("Saving encodings to encodings.pickle ...")
            data = {"encodings": known_encodings, "names": known_names}
            f = open(constants.ENCODINGS_PATH, "wb")
            f.write(pickle.dumps(data))
            f.close()
            print("Encodings have been saved successfully.")
            self.DONE_SIGNAL.emit(True)
        else:
            print("No images to train.")
            if os.path.exists(constants.ENCODINGS_PATH):
                os.remove(constants.ENCODINGS_PATH)
        print("\n [INFO] {0} faces trained.".format(len(known_names)))


class TrainerDlg(QDialog):
    """Setup Progress Bar Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create an instance of the GUI
        self.ui = TrainerUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
