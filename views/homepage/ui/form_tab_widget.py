from PySide6 import QtWidgets, QtCore


class FormTabWidget(QtWidgets.QTabWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # form tab widget setup code...
        # left tab menus
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(
            self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(size_policy)
        self.setObjectName("formTabWidget")
