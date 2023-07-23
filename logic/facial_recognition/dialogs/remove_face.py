import os
import shutil
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog
from shared import constants
from shared.message_prompts import show_info_messagebox


class RemoveFaceUI(object):
    """
    Creation for Remove Face UI
    """

    def __init__(self):
        self.path = None
        self.name_list = None
        self.horizontalLayout = None
        self.cancel_btn = None
        self.remove_face_btn = None
        self.remove_face_title_label = None
        self.verticalLayout = None
        self.window = None
        self.count = 0

    def setupUi(self, remove_face):
        """
        Used for setup when calling the RemoveFaceDlg Class
        :param remove_face:
        """
        self.window = remove_face
        remove_face.setObjectName("remove_face")
        remove_face.resize(180, 60)
        self.verticalLayout = QtWidgets.QVBoxLayout(remove_face)
        self.verticalLayout.setObjectName("verticalLayout")
        self.remove_face_title_label = QtWidgets.QLabel(remove_face)
        self.remove_face_title_label.setText("remove_face_title")
        self.verticalLayout.addWidget(self.remove_face_title_label)

        self.name_list = QtWidgets.QListWidget(remove_face)
        self.name_list.setObjectName("name_list")

        # Path for face image database
        for folder in os.listdir(constants.IMAGE_PATH):
            self.name_list.addItem(folder)

        self.verticalLayout.addWidget(self.name_list)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.remove_face_btn = QtWidgets.QPushButton(remove_face)
        self.remove_face_btn.setObjectName("remove_face_btn")
        self.remove_face_btn.clicked.connect(self.remove_face_prompt)
        self.horizontalLayout.addWidget(self.remove_face_btn)

        self.cancel_btn = QtWidgets.QPushButton(remove_face)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self.window.close)
        self.horizontalLayout.addWidget(self.cancel_btn)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.translate_ui(remove_face)
        QtCore.QMetaObject.connectSlotsByName(remove_face)

    def remove_face_prompt(self):
        """
        Method runs when user selects face in the list to delete.
        Removes the folder and all images inside.
        """
        selected_face = constants.IMAGE_PATH + self.name_list.currentItem().text()
        shutil.rmtree(selected_face)
        show_info_messagebox(
            "Face Removed.")
        self.window.close()

    def translate_ui(self, remove_face):
        """
        Automatic Translation Locale
        :param remove_face:
        """
        _translate = QtCore.QCoreApplication.translate
        remove_face.setWindowTitle(_translate("remove_face", "Remove Face"))
        self.remove_face_title_label.setText(
            _translate("remove_face_title", "Select Name:"))
        self.remove_face_btn.setText(_translate("remove_face_btn", "Remove"))
        self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class RemoveFaceDlg(QDialog):
    """Setup Remove Face Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create an instance of the GUI
        self.ui = RemoveFaceUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
