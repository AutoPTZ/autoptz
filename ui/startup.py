from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
import sys

from ui.homepage.homepage import Ui_AutoPTZ


class AutoPTZ(QtWidgets.QMainWindow, Ui_AutoPTZ):
    def __init__(self, parent=None):
        super(AutoPTZ, self).__init__(parent)

        self.setupUi(self)


def main():
    app = QApplication(sys.argv)
    form = AutoPTZ()
    form.show()
    app.exec_()


if __name__ == '__main__':
    main()
