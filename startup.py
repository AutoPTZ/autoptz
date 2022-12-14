import sys
from PySide6.QtWidgets import QApplication, QSystemTrayIcon
from PySide6.QtGui import QIcon, QPixmap

from views.homepage.main_window import AutoPTZ_MainWindow
import shared.constants as constants


def main():
    """
    Starts the AutoPTZ Application
    """
    app = QApplication(sys.argv)
    window = AutoPTZ_MainWindow()
    window.setWindowIcon(QIcon(constants.ICON_PNG))
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
