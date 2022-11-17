import os
import cv2
import face_recognition
import pickle
from imutils import paths
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog


class TrainerUI(object):
    trainer_request_update = QtCore.pyqtSignal(int)
    def __init__(self):
        self.training_progress_bar_title = None
        self.verticalLayout = None
        self.window = None
        self.progress_bar_line = None
        # self.horizontalLayout = None
        # self.cancel_btn = None
        # self.confirm_btn = None
        self.count = 0

    def setMaximumVal(self, value):
        print('set max')
        self.progress_bar_line.setMaximum(value)

    def setCurrentVal(self, value):
        print('set current')
        self.progress_bar_line.setValue(value)

    def setupUi(self, training_progress_bar):
        self.window = training_progress_bar
        training_progress_bar.setObjectName("training_progress_bar")
        training_progress_bar.resize(200, 120)
        self.verticalLayout = QtWidgets.QVBoxLayout(training_progress_bar)
        self.verticalLayout.setObjectName("verticalLayout")
        self.training_progress_bar_title = QtWidgets.QLabel(training_progress_bar)
        self.training_progress_bar_title.setText("training_progress_bar_title")
        self.verticalLayout.addWidget(self.training_progress_bar_title)

        self.progress_bar_line = QtWidgets.QProgressBar(training_progress_bar)
        self.progress_bar_line.setObjectName("progress_bar_line")
        self.verticalLayout.addWidget(self.progress_bar_line)

        print("a")
        self.trainer = TrainerThread()
        self.trainer_thread = QtCore.QThread()
        print("2")
        self.trainer.MAX_VALUE_SIGNAL.connect(self.setMaximumVal)
        self.trainer.CURRENT_VALUE_SIGNAL.connect(self.setCurrentVal)
        print("3")
        #self.trainer_request_update.connect(self.trainer.train_face)
        print("4")
        self.trainer.moveToThread(self.trainer_thread)
        print("5")
        self.trainer_thread.start()
        print("6")

        # self.progress_bar_line.setValue(10)
        # self.progress_bar_line.setMaximum(100)

        # maybe add cancel btn on bottom
        # self.horizontalLayout = QtWidgets.QHBoxLayout()
        # self.horizontalLayout.setObjectName("horizontalLayout")
        # self.confirm_btn = QtWidgets.QPushButton(reset_database)
        # self.confirm_btn.setObjectName("confirm_btn")
        # self.confirm_btn.clicked.connect(self.reset_database_prompt)
        # self.horizontalLayout.addWidget(self.confirm_btn)
        #
        # self.cancel_btn = QtWidgets.QPushButton(reset_database)
        # self.cancel_btn.setObjectName("cancel_btn")
        # self.cancel_btn.clicked.connect(self.window.close)
        # self.horizontalLayout.addWidget(self.cancel_btn)
        #
        # self.verticalLayout.addLayout(self.horizontalLayout)

        self.translate_ui(training_progress_bar)
        QtCore.QMetaObject.connectSlotsByName(training_progress_bar)

    def translate_ui(self, training_progress_bar):
        _translate = QtCore.QCoreApplication.translate
        training_progress_bar.setWindowTitle(_translate("training_progress_bar", "Training Model"))
        self.training_progress_bar_title.setText(_translate("training_progress_bar_title", "Looking at Images Folder"))
        # self.confirm_btn.setText(_translate("confirm_btn", "Confirm"))
        # self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class TrainerThread(QtCore.QObject):
    MAX_VALUE_SIGNAL = QtCore.pyqtSignal(int)
    CURRENT_VALUE_SIGNAL = QtCore.pyqtSignal(int)
    face_recognition = face_recognition.FaceRec()

    @QtCore.pyqtSlot(int)
    def train_face(self):
        print("\n [INFO] Training faces. It will take a few minutes. Please Wait ...")
        # Image path for face image database
        image_path = '../logic/facial_tracking/images/'
        encodings_path = '../logic/facial_tracking/trainer/encodings.pickle'

        imagePaths = list(paths.list_images(image_path))
        knownEncodings = []
        knownNames = []
        self.MAX_VALUE_SIGNAL.emit(len(imagePaths))
        self.CURRENT_VALUE_SIGNAL.emit(0)
        # self.progress_bar_ui.ui.progress_bar_line.setMaximum(len(imagePaths))
        # self.progress_bar_ui.ui.progress_bar_line.setValue(0)
        if os.listdir(image_path):
            # loop over the image paths
            for (i, imagePath) in enumerate(imagePaths):
                # extract the person name from the image path
                print(f"Processing {i + 1} of {len(imagePaths)}")
                self.CURRENT_VALUE_SIGNAL.emit(i + 1)
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
            # self.progress_bar_ui.close()
            print("Encodings have been saved successfully.")
        else:
            print("No images to train.")
            if os.path.exists(encodings_path):
                os.remove(encodings_path)
        # if show_message_box:
        #     show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(knownNames)))
        print("\n [INFO] {0} faces trained.".format(len(knownNames)))


class TrainerDlg(QDialog):
    """Setup Progress Bar Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create an instance of the GUI
        self.ui = TrainerUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)

# class Trainer:
#     face_recognition = face_recognition.FaceRec()
#
#     def __init__(self):
#         self.progress_bar_ui = None
#
#     def train_face(self, show_message_box):
#         # if show_message_box:
#         #     show_info_messagebox("It will take a few minutes.\n Please Wait ...")
#         print("\n [INFO] Training faces. It will take a few minutes. Please Wait ...")
#         self.progress_bar_ui = ProgressBarDlg()
#         # Image path for face image database
#         image_path = '../logic/facial_tracking/images/'
#         encodings_path = '../logic/facial_tracking/trainer/encodings.pickle'
#
#         imagePaths = list(paths.list_images(image_path))
#         knownEncodings = []
#         knownNames = []
#
#         # self.progress_bar_ui.ui.progress_bar_line.setMaximum(len(imagePaths))
#         # self.progress_bar_ui.ui.progress_bar_line.setValue(0)
#         #
#         # self.progress_bar_ui.show()
#         if os.listdir(image_path):
#             # loop over the image paths
#             for (i, imagePath) in enumerate(imagePaths):
#                 # extract the person name from the image path
#                 print(f"Processing {i + 1} of {len(imagePaths)}")
#                 self.progress_bar_ui.ui.progress_bar_line.setValue(i+1)
#                 name = imagePath.split(os.path.sep)[-2]
#                 image = cv2.imread(imagePath)
#                 rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#                 boxes = Trainer.face_recognition.face_locations(rgb, model='cnn')
#                 encodings = Trainer.face_recognition.face_encodings(rgb, boxes)
#                 for encoding in encodings:
#                     knownEncodings.append(encoding)
#                     knownNames.append(name)
#             print("Saving encodings to encodings.pickle ...")
#             data = {"encodings": knownEncodings, "names": knownNames}
#             f = open(encodings_path, "wb")
#             f.write(pickle.dumps(data))
#             f.close()
#             # self.progress_bar_ui.close()
#             print("Encodings have been saved successfully.")
#         else:
#             print("No images to train.")
#             if os.path.exists(encodings_path):
#                 os.remove(encodings_path)
#         # if show_message_box:
#         #     show_info_messagebox("{0} faces trained.\nOpening Basic Recognition Software".format(len(knownNames)))
#         print("\n [INFO] {0} faces trained.".format(len(knownNames)))
#         return
#
