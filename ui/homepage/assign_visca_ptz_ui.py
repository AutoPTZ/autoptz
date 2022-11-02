from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog

from ui.homepage.move_visca_ptz import ViscaPTZ


class AssignPTZUI(object):
    def __init__(self):
        self.cancel_btn = None
        self.assign_btn = None
        self.usable_camera_list = None
        self.assign_title_label = None
        self.verticalLayout = None
        self.ptz_id = None
        self.window = None
        self.camera_list = None
        self.assigned_list = None

    def setupUi(self, assign_ptz, camera_list, assigned_list, ptz_id):
        self.window = assign_ptz
        self.camera_list = camera_list
        self.assigned_list = assigned_list
        self.ptz_id = ptz_id
        assign_ptz.setObjectName("assign_ptz")
        assign_ptz.resize(270, 150)
        self.verticalLayout = QtWidgets.QVBoxLayout(assign_ptz)
        self.verticalLayout.setObjectName("verticalLayout")
        self.assign_title_label = QtWidgets.QLabel(assign_ptz)

        self.assign_title_label.setText("Assign PTZ " + ptz_id)
        self.verticalLayout.addWidget(self.assign_title_label)
        self.usable_camera_list = QtWidgets.QListWidget(assign_ptz)
        self.usable_camera_list.setObjectName("usable_camera_list")
        for item in camera_list:
            self.usable_camera_list.addItem(item.objectName())
        self.verticalLayout.addWidget(self.usable_camera_list)
        self.assign_btn = QtWidgets.QPushButton(assign_ptz)
        self.assign_btn.setObjectName("assign_btn")

        self.assign_btn.clicked.connect(self.assign_ptz)
        self.verticalLayout.addWidget(self.assign_btn)
        self.cancel_btn = QtWidgets.QPushButton(assign_ptz)
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self.window.close)
        self.verticalLayout.addWidget(self.cancel_btn)

        self.translate_ui(assign_ptz)
        QtCore.QMetaObject.connectSlotsByName(assign_ptz)

    def assign_ptz(self):
        print("Assigning PTZ")
        camera_widget = self.camera_list[self.usable_camera_list.currentRow()]
        camera_widget.image_processor_thread.set_ptz_tracker(ViscaPTZ(device_id=self.ptz_id))
        self.assigned_list.append(self.ptz_id)
        self.assigned_list.append(camera_widget)

        self.window.close()

    def translate_ui(self, assign_ptz):
        _translate = QtCore.QCoreApplication.translate
        assign_ptz.setWindowTitle(_translate("assign_ptz", "Assign PTZ"))
        self.assign_btn.setText(_translate("assign_ptz", "Assign"))
        self.cancel_btn.setText(_translate("assign_ptz", "Cancel"))


class AssignViscaPTZDlg(QDialog):
    """Assign PTZ to Serial Camera dialog."""

    def __init__(self, parent=None, camera_list=None, assigned_list=None, ptz_id=''):
        super().__init__(parent)
        # Create an instance of the GUI

        if assigned_list is None:
            assigned_list = []
        if camera_list is None:
            camera_list = []
        self.ui = AssignPTZUI()
        # Run the .setupUi() method to show the GUI
        self.ui.setupUi(self, camera_list=camera_list, assigned_list=assigned_list, ptz_id=ptz_id)
