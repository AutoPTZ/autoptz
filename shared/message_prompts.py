from PyQt6.QtWidgets import QMessageBox


def show_critical_messagebox(window_title, critical_message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    # setting message for Message Box
    msg.setText(critical_message)
    # setting Message box window title
    msg.setWindowTitle(window_title)
    # declaring buttons on Message Box
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    # start the app
    msg.exec()


def show_info_messagebox(info_message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Information)
    # setting message for Message Box
    msg.setText(info_message)
    # setting Message box window title
    msg.setWindowTitle("Information")
    # declaring buttons on Message Box
    msg.setStandardButtons(QMessageBox.StandardButton.Ok)
    # start the app
    msg.exec()
