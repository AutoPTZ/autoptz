import re
from visca_over_ip import CachingCamera
from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import QDialog

from shared.message_prompts import show_critical_messagebox


class AssignNetworkPTZIU(object):
    """
    Creation for Assign Network VISCA PTZ UI
    """

    def __init__(self):
        self.CachingCamera = None
        self.port_line = None
        self.horizontalLayout = None
        self.cancel_btn = None
        self.submit = None
        self.camera_widget = None
        self.allow_network_control = None
        self.verticalLayout = None
        self.window = None
        self.count = 0

    def setupUi(self, assign_net_ptz, camera_widget):
        """
        Used for setup when calling the AssignNetworkPTZDlg Class
        :param assign_net_ptz:
        :param camera_widget:
        """
        self.window = assign_net_ptz
        self.camera_widget = camera_widget
        assign_net_ptz.setObjectName("assign_net_ptz")
        assign_net_ptz.resize(300, 80)
        self.verticalLayout = QtWidgets.QVBoxLayout(assign_net_ptz)
        self.verticalLayout.setObjectName("verticalLayout")
        self.allow_network_control = QtWidgets.QLabel(assign_net_ptz)
        self.allow_network_control.setText("allow_network_control")
        self.verticalLayout.addWidget(self.allow_network_control)

        self.port_line = QtWidgets.QLineEdit(assign_net_ptz)
        self.port_line.setPlaceholderText("port_line")
        self.verticalLayout.addWidget(self.port_line)

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
        """
        Attempts to connect to network ptz by IP address and Port Number (default port is 52381)
        If it is successful the window closes.
        If not then a critical message box will appear, advising the user to check their camera settings.
        """
        ip_address = re.findall(r'(?:\d{1,3}\.)+\d{1,3}', self.camera_widget.objectName())
        print(ip_address, self.camera_widget.objectName())
        try:
            if self.port_line.text().strip() == "":
                camera_control = CachingCamera(ip=ip_address[0])
            else:
                camera_control = CachingCamera(ip=ip_address[0], port=self.port_line.text().strip())
            print("camera control started for " + ip_address[0])
            self.camera_widget.set_ptz(control=camera_control)
        except Exception as e:
            print(e)
            show_critical_messagebox(window_title="VISCA Camera Control",
                                     critical_message=f"Could not connect to {ip_address[0]}\nPlease check if VISCA "
                                                      "is enabled in your camera settings.")
        self.window.close()

    def translate_ui(self, add_face):
        """
        Automatic Translation Locale
        :param add_face:
        """
        _translate = QtCore.QCoreApplication.translate
        add_face.setWindowTitle(_translate("assign_net_ptz", "Assign Network PTZ"))
        self.allow_network_control.setText(
            _translate("allow_network_control", "VISCA Login for " + self.camera_widget.objectName() + ":"))
        self.port_line.setPlaceholderText(_translate("port_line", "Enter Port (Default=52381)"))
        self.submit.setText(_translate("submit", "Submit"))
        self.cancel_btn.setText(_translate("cancel_btn", "Cancel"))


class AssignNetworkPTZDlg(QDialog):
    """Run Assign Network Visca PTZ Dialog"""

    def __init__(self, parent=None, camera=None):
        super().__init__(parent)

        # Create an instance of the GUI
        self.ui = AssignNetworkPTZIU()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self, camera_widget=camera)
