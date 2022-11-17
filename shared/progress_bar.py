from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog


class ProgressBarUI(object):
    def __init__(self):
        self.training_progress_bar_title = None
        self.verticalLayout = None
        self.window = None
        self.progress_bar_line = None
        # self.horizontalLayout = None
        # self.cancel_btn = None
        # self.confirm_btn = None
        self.count = 0

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

    # submit btn
    # def reset_database_prompt(self):
    #     if self.confirm_line.text().strip() == "":
    #         return
    #     else:
    #         # check if phrase is correct, if so delete all images + trainer.yml
    #         if self.confirm_line.text().strip() == 'RESET ALL':
    #             image_path = '../logic/facial_tracking/images/'
    #             encodings_path = '../logic/facial_tracking/trainer/encodings.pickle'
    #             if os.path.exists(image_path):
    #                 shutil.rmtree(image_path)
    #             if os.path.exists(encodings_path):
    #                 os.remove(encodings_path)
    #             show_critical_messagebox(window_title="Reset Database",
    #                                      critical_message="All stored faces and models have been removed.")
    #             self.window.close()
    #         else:
    #             show_critical_messagebox(window_title="Reset Database",
    #                                      critical_message="Incorrect Phrase.\nPlease try again!")
    #             return



    def translate_ui(self, training_progress_bar):
        _translate = QtCore.QCoreApplication.translate
        training_progress_bar.setWindowTitle(_translate("training_progress_bar", "Training Model"))
        self.training_progress_bar_title.setText(_translate("training_progress_bar_title", "Looking at Images Folder"))
        # self.confirm_btn.setText(_translate("confirm_btn", "Confirm"))
        # self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class ProgressBarDlg(QDialog):
    """Setup Progress Bar Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Create an instance of the GUI
        self.ui = ProgressBarUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
