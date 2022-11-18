from PyQt6.QtWidgets import QApplication
import sys

from ui.homepage.main_window import AutoPTZ_MainWindow


# class AutoPTZ(QtWidgets.QMainWindow, Ui_AutoPTZ):
#     def __init__(self, parent=None):
#         super(AutoPTZ, self).__init__(parent)
#         self.setupUi(self)


def main():
    app = QApplication(sys.argv)
    window = AutoPTZ_MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
