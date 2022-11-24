import os
import shutil
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog

from shared import constants
from shared.message_prompts import show_critical_messagebox


class ResetDatabaseUI(object):
    """
    Creation for Reset Database UI
    """
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
        """
        Used for setup when calling the ResetDatabaseDlg Class
        :param reset_database:
        """
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
        """
        Methods that checks what the user inputs in the dialog.
        Checks to see if the user enters "RESET ALL" exactly to remove all trainer and stored image files.
        :return:
        """
        if self.confirm_line.text().strip() == "":
            return
        else:
            # check if phrase is correct, if so delete all images + trainer.yml
            if self.confirm_line.text().strip() == 'RESET ALL':
                if os.path.exists(constants.IMAGE_PATH):
                    shutil.rmtree(constants.IMAGE_PATH)
                if os.path.exists(constants.ENCODINGS_PATH):
                    os.remove(constants.ENCODINGS_PATH)
                show_critical_messagebox(window_title="Reset Database",
                                         critical_message="All stored faces and models have been removed.")
                self.window.close()
            else:
                show_critical_messagebox(window_title="Reset Database",
                                         critical_message="Incorrect Phrase.\nPlease try again!")
                return

    def translate_ui(self, reset_database):
        """
        Automatic Translation Locale
        :param reset_database:
        """
        _translate = QtCore.QCoreApplication.translate
        reset_database.setWindowTitle(_translate("reset_database", "Reset Database"))
        self.reset_database_title_label.setText(_translate("reset_database_title", "Type 'RESET ALL' to confirm"))
        self.confirm_btn.setText(_translate("confirm_btn", "Confirm"))
        self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class ResetDatabaseDlg(QDialog):
    """Setup Reset Database Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create an instance of the GUI

        self.ui = ResetDatabaseUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
