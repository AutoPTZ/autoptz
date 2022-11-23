import os
import re
from sensecam_control import onvif_control
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog

from ui.shared.message_prompts import show_critical_messagebox, show_info_messagebox


class AssignNetworkPTZIU(object):
    def __init__(self):
        self.username_line = None
        self.password_line = None
        self.horizontalLayout = None
        self.cancel_btn = None
        self.submit = None
        self.camera = None
        self.allow_network_control = None
        self.verticalLayout = None
        self.window = None
        self.count = 0

    def setupUi(self, assign_net_ptz, camera):
        self.window = assign_net_ptz
        self.camera = camera
        assign_net_ptz.setObjectName("assign_net_ptz")
        assign_net_ptz.resize(300, 80)
        self.verticalLayout = QtWidgets.QVBoxLayout(assign_net_ptz)
        self.verticalLayout.setObjectName("verticalLayout")
        self.allow_network_control = QtWidgets.QLabel(assign_net_ptz)
        self.allow_network_control.setText("allow_network_control")
        self.verticalLayout.addWidget(self.allow_network_control)

        self.username_line = QtWidgets.QLineEdit(assign_net_ptz)
        self.username_line.setPlaceholderText("username_line")
        self.verticalLayout.addWidget(self.username_line)

        self.password_line = QtWidgets.QLineEdit(assign_net_ptz)
        self.password_line.setPlaceholderText("password_line")
        self.verticalLayout.addWidget(self.password_line)

        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.submit = QtWidgets.QPushButton(assign_net_ptz)
        self.submit.setObjectName("submit")
        self.submit.clicked.connect(self.assign_net_ptz_prompt)
        self.horizontalLayout.addWidget(self.submit)

        self.cancel_btn = QtWidgets.QPushButton(assign_net_ptz)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self.window.close)
        self.horizontalLayout.addWidget(self.cancel_btn)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.translate_ui(assign_net_ptz)
        QtCore.QMetaObject.connectSlotsByName(assign_net_ptz)

    def assign_net_ptz_prompt(self):
        try:
            ip_address = re.findall(r'(?:\d{1,3}\.)+(?:\d{1,3})', self.camera.objectName())
            camera_control = onvif_control.CameraControl(ip_address[0], self.username_line.text().strip(),
                                                         self.password_line.text().strip())
            # camera_control.setDaemon(True)
            camera_control.camera_start()
            # camera_control.start()
            print("camera control started for " + ip_address[0])
            self.camera.image_processor_thread.set_ptz_controller(control=camera_control, isONVIF=True)
            self.camera.image_processor_thread.set_ptz_ready("ready")
            self.window.close()
        except:
            show_critical_messagebox(window_title="ONVIF Camera Control",
                                     critical_message="Username or password is incorrect.\nPlease check if ONVIF "
                                                      "is enabled in your camera settings.")

    def translate_ui(self, add_face):
        _translate = QtCore.QCoreApplication.translate
        add_face.setWindowTitle(_translate("assign_net_ptz", "Assign Network PTZ"))
        self.allow_network_control.setText(
            _translate("allow_network_control", "ONVIF Login for " + self.camera.objectName() + ":"))
        self.username_line.setPlaceholderText(_translate("username_line", "Enter Username (Optional)"))
        self.password_line.setPlaceholderText(_translate("password_line", "Enter Password (Optional)"))
        self.submit.setText(_translate("submit", "Submit"))
        self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class AssignNetworkPTZDlg(QDialog):
    """Setup Add Face Dialog"""

    def __init__(self, parent=None, camera=None):
        super().__init__(parent)

        # Create an instance of the GUI
        self.ui = AssignNetworkPTZIU()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self, camera=camera)
