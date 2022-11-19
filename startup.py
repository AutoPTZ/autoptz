import sys
from PyQt6.QtWidgets import QApplication

from views.homepage.main_window import AutoPTZ_MainWindow


def main():
    app = QApplication(sys.argv)
    window = AutoPTZ_MainWindow()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
