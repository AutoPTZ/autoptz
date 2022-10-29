import os
import shutil

import cv2
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog

from ui.shared.message_prompts import show_critical_messagebox, show_info_messagebox

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")


class ResetDatabaseUI(object):
    def __init__(self):
        self.confirm_line = None
        self.horizontalLayout = None
        self.cancel_btn = None
        self.confirm_btn = None
        self.reset_database_title_label = None
        self.verticalLayout = None
        self.window = None
        self.count = 0

    def setupUi(self, reset_database):
        self.window = reset_database
        reset_database.setObjectName("reset_database")
        reset_database.resize(150, 60)
        self.verticalLayout = QtWidgets.QVBoxLayout(reset_database)
        self.verticalLayout.setObjectName("verticalLayout")
        self.reset_database_title_label = QtWidgets.QLabel(reset_database)
        self.reset_database_title_label.setText("reset_database_title")
        self.verticalLayout.addWidget(self.reset_database_title_label)

        self.confirm_line = QtWidgets.QLineEdit(reset_database)
        self.confirm_line.setObjectName("confirm_line")
        self.verticalLayout.addWidget(self.confirm_line)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.confirm_btn = QtWidgets.QPushButton(reset_database)
        self.confirm_btn.setObjectName("confirm_btn")
        self.confirm_btn.clicked.connect(self.reset_database_prompt)
        self.horizontalLayout.addWidget(self.confirm_btn)

        self.cancel_btn = QtWidgets.QPushButton(reset_database)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self.window.close)
        self.horizontalLayout.addWidget(self.cancel_btn)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.translate_ui(reset_database)
        QtCore.QMetaObject.connectSlotsByName(reset_database)

    def reset_database_prompt(self):
        if self.confirm_line.text().strip() == "":
            return
        else:
            # check if phrase is correct, if so delete all images + trainer.yml
            if self.confirm_line.text().strip() == 'RESET ALL':
                image_path = '../logic/facial_tracking/images/'
                trainer_path = '../logic/facial_tracking/trainer/trainer.yml'
                if os.path.exists(image_path):
                    shutil.rmtree(image_path)
                if os.path.exists(trainer_path):
                    os.remove(trainer_path)
                show_critical_messagebox(window_title="Reset Database",
                                         critical_message="All stored faces and models have been removed.")
                self.window.close()
            else:
                show_critical_messagebox(window_title="Reset Database",
                                         critical_message="Incorrect Phrase.\nPlease try again!")
                return

    def translate_ui(self, reset_database):
        _translate = QtCore.QCoreApplication.translate
        reset_database.setWindowTitle(_translate("reset_database", "Reset Database"))
        self.reset_database_title_label.setText(_translate("reset_database_title", "Type 'RESET ALL' to confirm"))
        self.confirm_btn.setText(_translate("confirm_btn", "Confirm"))
        self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class ResetDatabaseDlg(QDialog):
    """Setup Add Face Dialog"""

    def __init__(self, parent=None, camera=None):
        super().__init__(parent)
        # Create an instance of the GUI

        if camera is None:
            camera = ''
        self.ui = ResetDatabaseUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
