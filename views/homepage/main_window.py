import os
from threading import Lock
from functools import partial
from PySide6 import QtCore, QtWidgets
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import QMainWindow

import watchdog.events
import watchdog.observers
import shared.constants as constants
from logic.camera_search.search_ndi import get_ndi_sources
from libraries.visca.move_visca_ptz import ViscaPTZ
from logic.camera_search.get_serial_cameras import COMPorts
from shared.message_prompts import show_info_messagebox
from shared.watch_trainer_directory import WatchTrainer
from views.functions.show_dialogs_ui import ShowDialog
from views.functions.assign_network_ptz_ui import AssignNetworkPTZDlg
from views.homepage.flow_layout import FlowLayout
from views.homepage.ui.selected_cam_tab import SelectedCamPage
from views.widgets.camera_widget import CameraWidget
from views.homepage.ui.form_tab_widget import FormTabWidget


class AutoPTZ_MainWindow(QMainWindow):
    """
    Configures and Handles the AutoPTZ MainWindow UI
    """

    def __init__(self, *args, **kwargs):
        super(AutoPTZ_MainWindow, self).__init__(*args, **kwargs)
        self.lock = Lock()
        self.setup_ui()

    def setup_ui(self):
        self.setup_main_window()
        self.setup_central_widget()
        self.setup_form_tab_widget()
        self.setup_flow_layout()
        self.setup_menu_bar()
        self.setup_status_bar()
        self.setup_actions()
        self.setup_observers()
        self.translate_ui(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def setup_main_window(self):
        self.setObjectName("AutoPTZ")
        self.resize(200, 450)
        self.setAutoFillBackground(False)
        self.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.setDockNestingEnabled(False)

    def setup_central_widget(self):
        self.central_widget = QtWidgets.QWidget(self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(
            self.central_widget.sizePolicy().hasHeightForWidth())
        self.central_widget.setSizePolicy(size_policy)
        self.central_widget.setObjectName("central_widget")
        self.gridLayout = QtWidgets.QGridLayout(self.central_widget)
        self.gridLayout.setObjectName("gridLayout")
        self.setCentralWidget(self.central_widget)

    def setup_form_tab_widget(self):
        self.formTabWidget = FormTabWidget(self.central_widget)
        self.gridLayout.addWidget(self.formTabWidget, 0, 0, 3, 1)

        self.selectedCamPage = SelectedCamPage(self.formTabWidget)
        self.formTabWidget.addTab(self.selectedCamPage, "")

        # self.manualControlPage = ManualControlPage()
        # self.formTabWidget.addTab(self.manualControlPage, "")

    def setup_flow_layout(self):
        self.flowLayout = FlowLayout()
        self.gridLayout.addLayout(self.flowLayout, 0, 1, 1, 1)

    def setup_menu_bar(self):
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 705, 24))
        self.menubar.setObjectName("menubar")
        self.setMenuBar(self.menubar)

    def setup_status_bar(self):
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)

    def setup_actions(self):
        # Define your actions here
        pass

    def setup_observers(self):
        if os.path.exists(constants.TRAINER_PATH) is False:
            os.mkdir(constants.TRAINER_PATH)
        self.watch_trainer = WatchTrainer()
        observer = watchdog.observers.Observer()
        observer.schedule(self.watch_trainer,
                          path=constants.TRAINER_PATH, recursive=True)
        observer.start()

    def translate_ui(self, MainWindow):
        # Define your translate_ui method here
        pass
