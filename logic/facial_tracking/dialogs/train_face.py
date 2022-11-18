import os
import cv2
from libraries import face_recognition
import pickle
from imutils import paths
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog


class TrainerUI(object):
    def __init__(self):
        self.percent_for_bar = None
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
        self.percent_for_bar.setText(str(int(value/self.progress_bar_line.maximum() * 100)) + "%")
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

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.progress_bar_line = QtWidgets.QProgressBar(training_progress_bar)
        self.progress_bar_line.setObjectName("progress_bar_line")
        self.horizontalLayout.addWidget(self.progress_bar_line)

        self.percent_for_bar = QtWidgets.QLabel(training_progress_bar)
        self.percent_for_bar.setText('0%  ')
        self.horizontalLayout.addWidget(self.percent_for_bar)
        self.verticalLayout.addLayout(self.horizontalLayout)

        self.translate_ui(training_progress_bar)
        QtCore.QMetaObject.connectSlotsByName(training_progress_bar)
        self.trainer_thread.start()

    def translate_ui(self, training_progress_bar):
        _translate = QtCore.QCoreApplication.translate
        training_progress_bar.setWindowTitle(_translate("training_progress_bar", "Training Model"))


class TrainerThread(QtCore.QThread):
    MAX_VALUE_SIGNAL = QtCore.pyqtSignal(int)
    CURRENT_VALUE_SIGNAL = QtCore.pyqtSignal(int)
    CURRENT_TEXT_SIGNAL = QtCore.pyqtSignal(str)
    DONE_SIGNAL = QtCore.pyqtSignal(bool)
    face_recognition = face_recognition.FaceRec()

    def run(self):
        print("\n [INFO] Training faces. It will take a few minutes. Please Wait ...")
        # Image path for face image database
        image_path = '../logic/facial_tracking/images/'
        encodings_path = '../logic/facial_tracking/trainer/encodings.pickle'

        imagePaths = list(paths.list_images(image_path))
        knownEncodings = []
        knownNames = []
        self.MAX_VALUE_SIGNAL.emit(len(imagePaths))
        QtCore.QCoreApplication.processEvents()
        self.CURRENT_VALUE_SIGNAL.emit(0)
        QtCore.QCoreApplication.processEvents()
        if os.listdir(image_path):
            # loop over the image paths
            for (i, imagePath) in enumerate(imagePaths):
                # extract the person name from the image path
                print(f"Processing {i + 1} of {len(imagePaths)}")
                self.CURRENT_TEXT_SIGNAL.emit(f"Processing {i + 1} of {len(imagePaths)}")
                QtCore.QCoreApplication.processEvents()
                self.CURRENT_VALUE_SIGNAL.emit(i + 1)
                QtCore.QCoreApplication.processEvents()
                name = imagePath.split(os.path.sep)[-2]
                image = cv2.imread(imagePath)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                boxes = self.face_recognition.face_locations(rgb, model='cnn')
                encodings = self.face_recognition.face_encodings(rgb, boxes)
                for encoding in encodings:
                    knownEncodings.append(encoding)
                    knownNames.append(name)
            print("Saving encodings to encodings.pickle ...")
            data = {"encodings": knownEncodings, "names": knownNames}
            f = open(encodings_path, "wb")
            f.write(pickle.dumps(data))
            f.close()
            print("Encodings have been saved successfully.")
            self.DONE_SIGNAL.emit(True)
            QtCore.QCoreApplication.processEvents()
        else:
            print("No images to train.")
            if os.path.exists(encodings_path):
                os.remove(encodings_path)
        print("\n [INFO] {0} faces trained.".format(len(knownNames)))


class TrainerDlg(QDialog):
    """Setup Progress Bar Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create an instance of the GUI
        self.ui = TrainerUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
