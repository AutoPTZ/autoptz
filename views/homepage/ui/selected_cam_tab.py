import os
import shared.constants as constants
from PySide6 import QtWidgets, QtCore


class SelectedCamPage(QtWidgets.QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # selected camera page setup code...
        # auto tab menu
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(
            self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)
        self.setMinimumSize(QtCore.QSize(163, 0))
        self.setMaximumSize(QtCore.QSize(16777215, 428))
        self.setObjectName("selectedCamPage")
        self.formLayout = QtWidgets.QFormLayout(self)
        self.formLayout.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.formLayout.setFormAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.formLayout.setObjectName("formLayout")
        self.select_face_dropdown = QtWidgets.QComboBox(self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy.setHeightForWidth(
            self.select_face_dropdown.sizePolicy().hasHeightForWidth())
        self.select_face_dropdown.setSizePolicy(size_policy)
        self.select_face_dropdown.setObjectName("select_face_dropdown")
        self.select_face_dropdown.setEnabled(False)
        # self.select_face_dropdown.currentTextChanged.connect(
        #     self.selected_face_change)
        self.select_face_dropdown.addItem('')
        if os.path.isdir(constants.IMAGE_PATH):
            for folder in os.listdir(constants.IMAGE_PATH):
                self.select_face_dropdown.addItem(folder)
