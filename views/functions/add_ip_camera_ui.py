import cv2
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog
from shared.message_prompts import show_critical_messagebox


class AddIPSourceUI(object):
    def __init__(self):
        self.username_line = None
        self.password_line = None
        self.ip_address_line = None
        self.port_line = None
        self.horizontalLayout_user = None
        self.horizontalLayout_ip = None
        self.horizontalLayout_btn = None
        self.cancel_btn = None
        self.submit = None
        self.verticalLayout = None
        self.window = None
        self.count = 0

    def setupUi(self, connect_to_ip):
        self.window = connect_to_ip
        connect_to_ip.setObjectName("connect_to_ip")
        connect_to_ip.resize(300, 80)
        self.verticalLayout = QtWidgets.QVBoxLayout(connect_to_ip)
        self.verticalLayout.setObjectName("verticalLayout")

        self.horizontalLayout_user = QtWidgets.QHBoxLayout(connect_to_ip)
        self.verticalLayout.addLayout(self.horizontalLayout_user)
        self.username_line = QtWidgets.QLineEdit(connect_to_ip)
        self.username_line.setPlaceholderText("username_line")
        self.horizontalLayout_user.addWidget(self.username_line)
        self.password_line = QtWidgets.QLineEdit(connect_to_ip)
        self.password_line.setPlaceholderText("password_line")
        self.horizontalLayout_user.addWidget(self.password_line)

        self.horizontalLayout_ip = QtWidgets.QHBoxLayout(connect_to_ip)
        self.verticalLayout.addLayout(self.horizontalLayout_ip)
        self.ip_address_line = QtWidgets.QLineEdit(connect_to_ip)
        self.ip_address_line.setPlaceholderText("ip_address_line")
        self.horizontalLayout_ip.addWidget(self.ip_address_line)
        self.port_line = QtWidgets.QLineEdit(connect_to_ip)
        self.port_line.setPlaceholderText("port_line")
        self.horizontalLayout_ip.addWidget(self.port_line)

        self.horizontalLayout_btn = QtWidgets.QHBoxLayout()
        self.horizontalLayout_btn.setObjectName("horizontalLayout_btn")
        self.verticalLayout.addLayout(self.horizontalLayout_btn)
        self.submit = QtWidgets.QPushButton(connect_to_ip)
        self.submit.setObjectName("submit")
        self.submit.clicked.connect(self.connect_to_ip_prompt)
        self.horizontalLayout_btn.addWidget(self.submit)
        self.cancel_btn = QtWidgets.QPushButton(connect_to_ip)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self.window.close)
        self.horizontalLayout_btn.addWidget(self.cancel_btn)

        self.translate_ui(connect_to_ip)
        QtCore.QMetaObject.connectSlotsByName(connect_to_ip)

    def connect_to_ip_prompt(self):
        vcap = cv2.VideoCapture(
            f"rtsp://{self.username_line.text().strip()}:{self.password_line.text().strip()}@{self.ip_address_line.text().strip()}:{self.port_line.text().strip()}/live")
        print(f"{self.username_line.text().strip()}:{self.password_line.text().strip()}@{self.ip_address_line.text().strip()}:{self.port_line.text().strip()}")
        while True:
            ret, frame = vcap.read()
            if not ret:
                show_critical_messagebox(window_title="IP Camera Source",
                                         critical_message="Something went wrong.\nPlease try again!")
                break
            else:
                self.window.close()
            vcap.release()


    def translate_ui(self, connect_to_ip):
        _translate = QtCore.QCoreApplication.translate
        connect_to_ip.setWindowTitle(_translate("connect_to_ip", "Connect to IP Camera"))
        self.username_line.setPlaceholderText(_translate("username_line", "Enter Username"))
        self.password_line.setPlaceholderText(_translate("password_line", "Enter Password"))
        self.ip_address_line.setPlaceholderText(_translate("ip_address_line", "Enter IP Address"))
        self.port_line.setPlaceholderText(_translate("port_line", "Enter Port"))
        self.submit.setText(_translate("submit", "Submit"))
        self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class AddIPSourceDlg(QDialog):
    """Setup Add Face Dialog"""

    def __init__(self, parent=None):
        super().__init__(parent)

        # Create an instance of the GUI
        self.ui = AddIPSourceUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self)
