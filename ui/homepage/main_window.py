import os
import platform
import shutil
import watchdog.events
import watchdog.observers
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtMultimedia import QMediaDevices
from PyQt6.QtWidgets import QMainWindow

from logic.facial_tracking.dialogs.add_face import AddFaceDlg
from logic.facial_tracking.dialogs.remove_face import RemoveFaceDlg
from logic.facial_tracking.dialogs.reset_database import ResetDatabaseDlg
from logic.facial_tracking.dialogs.train_face import TrainerDlg
from ui.homepage.assign_network_ptz_ui import AssignNetworkPTZDlg
from logic.facial_tracking.move_visca_ptz import ViscaPTZ
from ui.homepage.assign_visca_ptz_ui import AssignViscaPTZDlg
from ui.homepage.flow_layout import FlowLayout
from logic.camera_search.get_serial_cameras import COMPorts
from logic.camera_search.search_ndi import get_ndi_sources
from shared.message_prompts import show_info_messagebox
from shared.watch_trainer_directory import WatchTrainer
from ui.widgets.camera_widget import CameraWidget
from ui.widgets.ndi_cam_widget import NDICameraWidget
import shared.constants as constants


class AutoPTZ_MainWindow(QMainWindow):
    def __init__(self, *args, **kwargs):
        # setting up the UI and QT Threading
        super(AutoPTZ_MainWindow, self).__init__(*args, **kwargs)
        self.threadpool = QtCore.QThreadPool()
        # self.threadpool.setMaxThreadCount(1)
        self.threadpool.maxThreadCount()

        # setting up main window
        self.setObjectName("AutoPTZ")
        self.resize(200, 450)
        self.setAutoFillBackground(False)
        self.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.setDockNestingEnabled(False)

        # base window widget
        self.central_widget = QtWidgets.QWidget(self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.central_widget.sizePolicy().hasHeightForWidth())
        self.central_widget.setSizePolicy(size_policy)
        self.central_widget.setObjectName("central_widget")
        self.gridLayout = QtWidgets.QGridLayout(self.central_widget)
        self.gridLayout.setObjectName("gridLayout")
        self.setCentralWidget(self.central_widget)

        # left tab menus
        self.formTabWidget = QtWidgets.QTabWidget(self.central_widget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.formTabWidget.sizePolicy().hasHeightForWidth())
        self.formTabWidget.setSizePolicy(size_policy)
        self.formTabWidget.setObjectName("formTabWidget")

        # auto tab menu
        self.selectedCamPage = QtWidgets.QWidget(self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.selectedCamPage.sizePolicy().hasHeightForWidth())
        self.selectedCamPage.setSizePolicy(size_policy)
        self.selectedCamPage.setMinimumSize(QtCore.QSize(163, 0))
        self.selectedCamPage.setMaximumSize(QtCore.QSize(16777215, 428))
        self.selectedCamPage.setObjectName("selectedCamPage")
        self.formLayout = QtWidgets.QFormLayout(self.selectedCamPage)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setObjectName("formLayout")
        self.select_face_dropdown = QtWidgets.QComboBox(self.selectedCamPage)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        size_policy.setHeightForWidth(self.select_face_dropdown.sizePolicy().hasHeightForWidth())
        self.select_face_dropdown.setSizePolicy(size_policy)
        self.select_face_dropdown.setObjectName("select_face_dropdown")
        self.select_face_dropdown.setEnabled(False)
        self.select_face_dropdown.currentTextChanged.connect(self.selected_face_change)
        self.select_face_dropdown.addItem('')
        if os.path.isdir(self.image_path):
            for folder in os.listdir(self.image_path):
                self.select_face_dropdown.addItem(folder)

        # assign VISCA PTZ to Serial Camera Source
        self.assign_network_ptz_btn = QtWidgets.QPushButton(self.selectedCamPage)
        self.assign_network_ptz_btn.setGeometry(QtCore.QRect(10, 380, 150, 32))
        self.assign_network_ptz_btn.setObjectName("assign_network_ptz_btn")
        self.assign_network_ptz_btn.hide()
        self.unassign_network_ptz_btn = QtWidgets.QPushButton(self.selectedCamPage)
        self.unassign_network_ptz_btn.setGeometry(QtCore.QRect(0, 380, 162, 32))
        self.unassign_network_ptz_btn.setObjectName("unassign_visca_ptz_btn")
        self.unassign_network_ptz_btn.hide()
        self.assign_network_ptz_btn.clicked.connect(self.assign_network_ptz_dlg)
        self.unassign_network_ptz_btn.clicked.connect(self.unassign_network_ptz)

        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.select_face_dropdown)
        self.enable_track = QtWidgets.QCheckBox(self.selectedCamPage)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        size_policy.setHeightForWidth(self.enable_track.sizePolicy().hasHeightForWidth())
        self.enable_track.setSizePolicy(size_policy)
        self.enable_track.setChecked(False)
        self.enable_track.setEnabled(False)
        self.enable_track.setAutoRepeat(False)
        self.enable_track.setAutoExclusive(False)
        self.enable_track.stateChanged.connect(self.config_enable_track)
        self.enable_track.setObjectName("enable_track")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.enable_track)
        self.select_face_label = QtWidgets.QLabel(self.selectedCamPage)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.select_face_label.sizePolicy().hasHeightForWidth())
        self.select_face_label.setSizePolicy(size_policy)
        self.select_face_label.setObjectName("select_face_label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.select_face_label)
        self.formTabWidget.addTab(self.selectedCamPage, "")

        # manual control tab menu
        self.manualControlPage = QtWidgets.QWidget()
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding)
        size_policy.setHeightForWidth(self.manualControlPage.sizePolicy().hasHeightForWidth())
        self.manualControlPage.setSizePolicy(size_policy)
        self.manualControlPage.setMinimumSize(QtCore.QSize(163, 0))
        self.manualControlPage.setMaximumSize(QtCore.QSize(16777215, 428))
        self.manualControlPage.setObjectName("manualControlPage")
        self.select_camera_label = QtWidgets.QLabel(self.manualControlPage)
        self.select_camera_label.setGeometry(QtCore.QRect(10, 30, 101, 21))
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.select_camera_label.sizePolicy().hasHeightForWidth())
        self.select_camera_label.setSizePolicy(size_policy)
        self.select_camera_label.setObjectName("select_camera_label")
        self.select_camera_dropdown = QtWidgets.QComboBox(self.manualControlPage)
        self.select_camera_dropdown.setGeometry(QtCore.QRect(9, 51, 151, 26))
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        size_policy.setHeightForWidth(self.select_camera_dropdown.sizePolicy().hasHeightForWidth())
        self.select_camera_dropdown.setSizePolicy(size_policy)
        self.select_camera_dropdown.setObjectName("select_camera_dropdown")
        self.select_camera_dropdown.addItem("")

        # add all the USB VISCA devices to the dropdown menu
        data_list = COMPorts.get_com_ports().data
        for port in data_list:
            if "USB" in port.description:
                print(port.device, port.description, data_list.index(port))
                self.select_camera_dropdown.addItem(port.device)

        self.select_camera_dropdown.currentTextChanged.connect(self.init_manual_control)

        # manual control buttons
        self.gridLayoutWidget = QtWidgets.QWidget(self.manualControlPage)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 100, 162, 131))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.controller_layout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.controller_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.controller_layout.setContentsMargins(0, 0, 0, 0)
        self.controller_layout.setObjectName("controllerLayout")
        self.down_right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.down_right_btn.sizePolicy().hasHeightForWidth())
        self.down_right_btn.setSizePolicy(size_policy)
        self.down_right_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_right_btn.setFlat(False)
        self.down_right_btn.setObjectName("down_right_btn")
        self.controller_layout.addWidget(self.down_right_btn, 2, 2, 1, 1)
        self.up_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.up_btn.sizePolicy().hasHeightForWidth())
        self.up_btn.setSizePolicy(size_policy)
        self.up_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_btn.setObjectName("up_btn")
        self.controller_layout.addWidget(self.up_btn, 0, 1, 1, 1)
        self.up_left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.up_left_btn.sizePolicy().hasHeightForWidth())
        self.up_left_btn.setSizePolicy(size_policy)
        self.up_left_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_left_btn.setObjectName("up_left_btn")
        self.controller_layout.addWidget(self.up_left_btn, 0, 0, 1, 1)
        self.left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.left_btn.sizePolicy().hasHeightForWidth())
        self.left_btn.setSizePolicy(size_policy)
        self.left_btn.setIconSize(QtCore.QSize(10, 10))
        self.left_btn.setObjectName("left_btn")
        self.controller_layout.addWidget(self.left_btn, 1, 0, 1, 1)
        self.down_left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.down_left_btn.sizePolicy().hasHeightForWidth())
        self.down_left_btn.setSizePolicy(size_policy)
        self.down_left_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_left_btn.setObjectName("down_left_btn")
        self.controller_layout.addWidget(self.down_left_btn, 2, 0, 1, 1)
        self.up_right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.up_right_btn.sizePolicy().hasHeightForWidth())
        self.up_right_btn.setSizePolicy(size_policy)
        self.up_right_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_right_btn.setObjectName("up_right_btn")
        self.controller_layout.addWidget(self.up_right_btn, 0, 2, 1, 1)
        self.right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.right_btn.sizePolicy().hasHeightForWidth())
        self.right_btn.setSizePolicy(size_policy)
        self.right_btn.setIconSize(QtCore.QSize(10, 10))
        self.right_btn.setObjectName("right_btn")
        self.controller_layout.addWidget(self.right_btn, 1, 2, 1, 1)
        self.down_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.down_btn.sizePolicy().hasHeightForWidth())
        self.down_btn.setSizePolicy(size_policy)
        self.down_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_btn.setObjectName("down_btn")
        self.controller_layout.addWidget(self.down_btn, 2, 1, 1, 1)
        self.home_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        size_policy.setHeightForWidth(self.home_btn.sizePolicy().hasHeightForWidth())
        self.home_btn.setSizePolicy(size_policy)
        self.home_btn.setIconSize(QtCore.QSize(10, 10))
        self.home_btn.setObjectName("home_btn")
        self.controller_layout.addWidget(self.home_btn, 1, 1, 1, 1)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.manualControlPage)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 240, 161, 32))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.zoom_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.zoom_layout.setContentsMargins(0, 0, 0, 0)
        self.zoom_layout.setObjectName("zoom_layout")
        self.zoom_in_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.zoom_in_btn.sizePolicy().hasHeightForWidth())
        self.zoom_in_btn.setSizePolicy(size_policy)
        self.zoom_in_btn.setObjectName("zoom_in_btn")
        self.zoom_layout.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.zoom_out_btn.sizePolicy().hasHeightForWidth())
        self.zoom_out_btn.setSizePolicy(size_policy)
        self.zoom_out_btn.setObjectName("zoom_out_btn")
        self.zoom_layout.addWidget(self.zoom_out_btn)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.manualControlPage)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 280, 161, 32))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.focus_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.focus_layout.setContentsMargins(0, 0, 0, 0)
        self.focus_layout.setObjectName("focus_layout")
        self.focus_plus_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.focus_plus_btn.sizePolicy().hasHeightForWidth())
        self.focus_plus_btn.setSizePolicy(size_policy)
        self.focus_plus_btn.setObjectName("focus_plus_btn")
        self.focus_layout.addWidget(self.focus_plus_btn)
        self.focus_minus_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.focus_minus_btn.sizePolicy().hasHeightForWidth())
        self.focus_minus_btn.setSizePolicy(size_policy)
        self.focus_minus_btn.setObjectName("focus_minus_btn")
        self.focus_layout.addWidget(self.focus_minus_btn)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.manualControlPage)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 320, 161, 32))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.menu_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.menu_layout.setContentsMargins(0, 0, 0, 0)
        self.menu_layout.setObjectName("menu_layout")
        self.menu_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.menu_btn.sizePolicy().hasHeightForWidth())
        self.menu_btn.setSizePolicy(size_policy)
        self.menu_btn.setObjectName("menu_btn")
        self.menu_layout.addWidget(self.menu_btn)
        self.reset_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.reset_btn.sizePolicy().hasHeightForWidth())
        self.reset_btn.setSizePolicy(size_policy)
        self.reset_btn.setObjectName("reset_btn")
        self.menu_layout.addWidget(self.reset_btn)

        # assign VISCA PTZ to Serial Camera Source
        self.assign_visca_ptz_btn = QtWidgets.QPushButton(self.manualControlPage)
        self.assign_visca_ptz_btn.setGeometry(QtCore.QRect(10, 380, 141, 32))
        self.assign_visca_ptz_btn.setObjectName("assign_visca_ptz_btn")
        self.assign_visca_ptz_btn.hide()
        self.unassign_visca_ptz_btn = QtWidgets.QPushButton(self.manualControlPage)
        self.unassign_visca_ptz_btn.setGeometry(QtCore.QRect(10, 380, 141, 32))
        self.unassign_visca_ptz_btn.setObjectName("unassign_visca_ptz_btn")
        self.unassign_visca_ptz_btn.hide()
        self.assign_visca_ptz_btn.clicked.connect(self.assign_visca_ptz_dlg)
        self.unassign_visca_ptz_btn.clicked.connect(self.unassign_visca_ptz)
        self.formTabWidget.addTab(self.manualControlPage, "")
        self.gridLayout.addWidget(self.formTabWidget, 0, 0, 3, 1)

        # enabled cameras view
        self.shown_cameras = QtWidgets.QWidget()
        self.flowLayout = FlowLayout()
        self.flowLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.shown_cameras.setLayout(self.flowLayout)
        self.shown_cameras.setSizePolicy(size_policy)
        self.gridLayout.addWidget(self.shown_cameras, 0, 1, 1, 1)

        # handling camera window sizing
        self.screen_width = QtWidgets.QApplication.desktop().screenGeometry().width()
        self.screen_height = QtWidgets.QApplication.desktop().screenGeometry().height()

        # Top Menu
        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 705, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSource = QtWidgets.QMenu(self.menubar)
        self.menuSource.setObjectName("menuSource")
        self.menuFacial_Recognition = QtWidgets.QMenu(self.menubar)
        self.menuFacial_Recognition.setObjectName("menuFacial_Recognition")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        self.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(self)
        self.statusbar.setObjectName("statusbar")
        self.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(self)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(self)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtWidgets.QAction(self)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionClose = QtWidgets.QAction(self)
        self.actionClose.setObjectName("actionClose")
        self.actionAdd_IP = QtWidgets.QAction(self)
        self.actionAdd_IP.setObjectName("actionAdd_IP")
        self.menuAdd_NDI = QtWidgets.QMenu(self)
        self.menuAdd_NDI.setObjectName("menuAdd_NDI")
        self.menuAdd_Hardware = QtWidgets.QMenu(self)
        self.menuAdd_Hardware.setObjectName("menuAdd_Hardware")
        self.actionEdit = QtWidgets.QAction(self)
        self.actionEdit.setObjectName("actionEdit")
        self.actionContact = QtWidgets.QAction(self)
        self.actionContact.setObjectName("actionContact")
        self.actionAbout = QtWidgets.QAction(self)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAdd_Face = QtWidgets.QAction(self)
        self.actionAdd_Face.setObjectName("actionAdd_Face")
        self.actionAdd_Face.triggered.connect(self.add_face)
        self.actionTrain_Model = QtWidgets.QAction(self)
        self.actionTrain_Model.setObjectName("actionTrain_Model")
        self.actionTrain_Model.triggered.connect(self.retrain_face)
        self.actionRemove_Face = QtWidgets.QAction(self)
        self.actionRemove_Face.setObjectName("actionRemove_Face")
        self.actionRemove_Face.triggered.connect(self.remove_face)
        self.actionReset_Database = QtWidgets.QAction(self)
        self.actionReset_Database.setObjectName("actionReset_Database")
        self.actionReset_Database.triggered.connect(self.reset_database)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSave_as)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClose)
        self.menuSource.addAction(self.actionAdd_IP)
        self.menuSource.addMenu(self.menuAdd_NDI)
        self.menuSource.addMenu(self.menuAdd_Hardware)
        self.menuSource.addSeparator()
        self.menuSource.addAction(self.actionEdit)
        self.menuFacial_Recognition.addAction(self.actionAdd_Face)
        self.menuFacial_Recognition.addAction(self.actionTrain_Model)
        self.menuFacial_Recognition.addAction(self.actionRemove_Face)
        self.menuFacial_Recognition.addSeparator()
        self.menuFacial_Recognition.addAction(self.actionReset_Database)
        self.menuHelp.addAction(self.actionAbout)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionContact)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSource.menuAction())
        self.menubar.addAction(self.menuFacial_Recognition.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        # other setup variables and methods
        self.getNDISourceList()
        self.getPhysicalSourcesList()
        self.assigned_ptz_camera = []
        self.serial_widget_list = []
        if os.path.exists(constants.TRAINER_PATH) is False:
            os.mkdir(constants.TRAINER_PATH)
        self.watch_trainer = WatchTrainer()
        observer = watchdog.observers.Observer()
        observer.schedule(self.watch_trainer, path=constants.TRAINER_PATH, recursive=True)
        observer.start()
        self.translateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def init_manual_control(self, device):
        """Initializing manual camera control. ONLY VISCA devices for now."""
        self.current_manual_device = ViscaPTZ(device_id=device)

        if device != "":
            # Enable Button Commands
            self.up_left_btn.clicked.connect(self.current_manual_device.move_left_up)
            self.up_btn.clicked.connect(self.current_manual_device.move_up)
            self.up_right_btn.clicked.connect(self.current_manual_device.move_right_up)
            self.left_btn.clicked.connect(self.current_manual_device.move_left)
            self.right_btn.clicked.connect(self.current_manual_device.move_right)
            self.down_left_btn.clicked.connect(self.current_manual_device.move_left_down)
            self.down_btn.clicked.connect(self.current_manual_device.move_down)
            self.down_right_btn.clicked.connect(self.current_manual_device.move_right_down)
            self.home_btn.clicked.connect(self.current_manual_device.move_home)
            self.zoom_in_btn.clicked.connect(self.current_manual_device.zoom_in)
            self.zoom_out_btn.clicked.connect(self.current_manual_device.zoom_out)
            self.menu_btn.clicked.connect(self.current_manual_device.menu)
            self.reset_btn.clicked.connect(self.current_manual_device.reset)
        else:
            # Disable Button Commands
            self.up_left_btn.disconnect()
            self.up_btn.disconnect()
            self.up_right_btn.disconnect()
            self.left_btn.disconnect()
            self.right_btn.disconnect()
            self.down_left_btn.disconnect()
            self.down_btn.disconnect()
            self.down_right_btn.disconnect()
            self.home_btn.disconnect()
            self.zoom_in_btn.disconnect()
            self.zoom_out_btn.disconnect()
            self.menu_btn.disconnect()
            self.reset_btn.disconnect()

        # shows button depending on if device has already been assigned to a camera source
        try:
            if device in self.assigned_ptz_camera:
                self.unassign_visca_ptz_btn.show()
            else:
                self.assign_visca_ptz_btn.show()
        except:
            self.assign_visca_ptz_btn.hide()
            self.unassign_visca_ptz_btn.hide()

    def assign_visca_ptz_dlg(self):
        """Launch the Assign VISCA PTZ to Camera Source dialog."""
        if not self.serial_widget_list or self.select_camera_dropdown.currentText() == "":
            print("Need to select or add a camera")
        else:
            dlg = AssignViscaPTZDlg(self, camera_list=self.serial_widget_list, assigned_list=self.assigned_ptz_camera,
                                    ptz_id=self.select_camera_dropdown.currentText())
            dlg.closeEvent = self.refreshViscaBtn
            dlg.exec()

    def unassign_visca_ptz(self):
        """Allow User to Unassign current VISCA PTZ device from Camera Source"""
        index = self.assigned_ptz_camera.index(self.select_camera_dropdown.currentText())

        camera = self.assigned_ptz_camera[index + 1]
        camera.image_processor.set_ptz_controller(None)
        self.assigned_ptz_camera.remove(camera)
        self.assigned_ptz_camera.remove(self.select_camera_dropdown.currentText())

        self.unassign_visca_ptz_btn.hide()
        self.assign_visca_ptz_btn.show()

    def assign_network_ptz_dlg(self):
        """Launch the Assign Network PTZ to Camera Source dialog."""
        if not self.current_selected_source:
            print("Need to select or add a camera")
        else:
            dlg = AssignNetworkPTZDlg(self, camera=self.current_selected_source)
            dlg.closeEvent = self.refreshOnvifBtn
            dlg.exec()

    def unassign_network_ptz(self):
        """Allow User to Unassign current Network PTZ device from Camera Source"""
        self.current_selected_source.image_processor.set_ptz_controller(control=None)
        self.current_selected_source.image_processor.set_ptz_ready("not ready")
        self.unassign_network_ptz_btn.hide()
        self.assign_network_ptz_btn.show()

    def add_face(self):
        """Launch the Add Face dialog based on the currently selected camera."""
        if self.flowLayout.count() == 0 or self.current_selected_source is None:
            show_info_messagebox("Please add and select a camera.")
        else:
            print("Opening Face Dialog")
            dlg = AddFaceDlg(self, camera=self.current_selected_source)
            dlg.closeEvent = self.update_face_selection
            dlg.exec()

    def update_face_selection(self, event):
        current_text_temp = self.select_face_dropdown.currentText()
        self.select_face_dropdown.clear()
        self.select_face_dropdown.addItem('')
        if os.path.exists(self.image_path):
            for folder in os.listdir(self.image_path):
                self.select_face_dropdown.addItem(folder)
            if self.select_face_dropdown.findText(current_text_temp) != -1:
                self.select_face_dropdown.setCurrentText(current_text_temp)

    @staticmethod
    def retrain_face():
        if not os.path.isdir(constants.IMAGE_PATH) or not os.listdir(constants.IMAGE_PATH):
            show_info_messagebox("No Faces to train.")
        else:
            TrainerDlg().show()

    def remove_face(self):
        """Launch the Remove Face dialog based on the currently selected camera."""
        if not os.path.isdir(constants.IMAGE_PATH) or not os.listdir(constants.IMAGE_PATH):
            show_info_messagebox("No Faces to remove.")
        else:
            current_len = len(os.listdir(constants.IMAGE_PATH))
            print("Opening Face Dialog")
            dlg = RemoveFaceDlg(self)
            dlg.closeEvent = self.update_face_selection
            dlg.exec()
            if not os.listdir(constants.IMAGE_PATH):
                if os.path.exists(constants.IMAGE_PATH):
                    shutil.rmtree(constants.IMAGE_PATH)
                if os.path.exists(constants.ENCODINGS_PATH):
                    os.remove(constants.ENCODINGS_PATH)
            elif current_len is not len(os.listdir(constants.IMAGE_PATH)):
                TrainerDlg().show()

    def reset_database(self):
        """Launch the Remove Face dialog based on the currently selected camera."""
        print("Opening Face Dialog")
        dlg = ResetDatabaseDlg(self)
        dlg.closeEvent = self.update_face_selection
        dlg.exec()

    def refreshViscaBtn(self, event):
        """Check is VISCA PTZ is assigned and change assignment button if so"""
        if self.select_camera_dropdown.currentText() in self.assigned_ptz_camera:
            self.unassign_visca_ptz_btn.show()
            self.assign_visca_ptz_btn.hide()
        else:
            self.assign_visca_ptz_btn.show()
            self.unassign_visca_ptz_btn.hide()

    def refreshOnvifBtn(self, event):
        """Check is Network PTZ is assigned and change assignment button if so"""
        if self.current_selected_source.image_processor.get_ptz_ready() == "ready":
            self.unassign_network_ptz_btn.show()
            self.assign_network_ptz_btn.hide()
        else:
            self.assign_network_ptz_btn.show()
            self.unassign_network_ptz_btn.hide()

    def selected_face_change(self):
        if self.current_selected_source is not None:
            if self.select_face_dropdown.currentText() == '':
                self.current_selected_source.image_processor.set_face(None)
                self.enable_track.setEnabled(False)
                self.enable_track.setChecked(False)
            else:
                self.current_selected_source.image_processor.set_face(self.select_face_dropdown.currentText())
                self.enable_track.setEnabled(True)
        else:
            self.enable_track.setEnabled(False)
            self.enable_track.setChecked(False)

    def config_enable_track(self):
        if self.current_selected_source is not None and self.current_selected_source.image_processor.is_track_enabled() and self.enable_track.isChecked():
            pass
        else:
            try:
                self.current_selected_source.image_processor.config_enable_track()
                self.enable_track.setChecked(self.current_selected_source.image_processor.is_track_enabled())
            except:
                self.enable_track.setChecked(False)

    def getPhysicalSourcesList(self):
        """Adds all camera sources to the physical source list"""
        available_cameras = QMediaDevices.videoInputs()
        index = 0
        for cam in available_cameras:
            if cam.description() != "NDI Video":
                if platform.system() == "Darwin":  # MacOS messes up the number scheme so this was the best fix
                    self.addPhysicalSource(source_number=index, source_name=cam.description())
                    index = 1 + index
                else:
                    self.addPhysicalSource(source_number=index, source_name=cam.description())
                    index = 1 + index

    def getNDISourceList(self):
        """Checks all NDI source in the network and adds them to the NDI source list"""
        source_list = get_ndi_sources()
        for i, s in enumerate(source_list):
            self.addNDISource(s)

    def addPhysicalSource(self, source_name, source_number):
        """Add selected Serial camera source from the menu to the camera grid"""
        camera_source = QtWidgets.QAction(source_name, self)
        camera_source.setCheckable(True)
        camera_source.triggered.connect(lambda: self.addCamera(source_number, menu_item=camera_source, ndi_source=None))
        self.menuAdd_Hardware.addAction(camera_source)

    def addNDISource(self, ndi_source_id):
        """Add selected NDI camera source from the menu to the camera grid"""
        ndi_source = QtWidgets.QAction(ndi_source_id.ndi_name, self)
        ndi_source.setCheckable(True)
        ndi_source.triggered.connect(lambda: self.addCamera(-1, ndi_source_id, ndi_source))
        self.menuAdd_NDI.addAction(ndi_source)

    def addCamera(self, source, ndi_source, menu_item):
        """Add NDI/Serial camera source from the menu to the camera grid"""
        camera_widget = QtWidgets.QWidget()

        if source == -1:
            # Make NDI Camera Widget
            camera = NDICameraWidget(self.screen_width // 3, self.screen_height // 3, ndi_source=ndi_source,
                                     aspect_ratio=True)
            camera.setObjectName('NDI Camera: ' + ndi_source.ndi_name)
            camera_widget.setObjectName('NDI Camera: ' + ndi_source.ndi_name)
            menu_item.disconnect()
            menu_item.triggered.connect(
                lambda: self.deleteCameraSource(source=-1, ndi_source=ndi_source, menu_item=menu_item, camera=camera,
                                                camera_widget=camera_widget))
        else:
            # Make Serial Camera Widget
            camera = CameraWidget(self.screen_width // 3, self.screen_height // 3, source, aspect_ratio=True)
            camera.setObjectName('Camera: ' + str(source + 1))
            camera_widget.setObjectName('Camera ' + str(source + 1))
            menu_item.disconnect()
            menu_item.triggered.connect(
                lambda: self.deleteCameraSource(source=source, menu_item=menu_item,
                                                ndi_source=None, camera=camera, camera_widget=camera_widget))
            self.serial_widget_list.append(camera)

        # create internal grid layout for camera
        camera_grid_layout = QtWidgets.QGridLayout()
        camera_grid_layout.setObjectName('Camera Grid: ' + str(camera))

        camera_grid_layout.addWidget(camera.get_video_frame(), 0, 0, 1, 1)
        camera_widget.setLayout(camera_grid_layout)

        select_cam_btn = QtWidgets.QPushButton("Select Camera")
        select_cam_btn.clicked.connect(lambda: self.selectCameraSource(camera=camera, select_cam_btn=select_cam_btn,
                                                                       unselect_cam_btn=unselect_cam_btn))
        unselect_cam_btn = QtWidgets.QPushButton("Unselect Camera")
        unselect_cam_btn.clicked.connect(
            lambda: self.unselectCameraSource(select_cam_btn=select_cam_btn, unselect_cam_btn=unselect_cam_btn))
        camera_grid_layout.addWidget(select_cam_btn, 1, 0, 1, 1)
        camera_grid_layout.addWidget(unselect_cam_btn, 2, 0, 1, 1)
        unselect_cam_btn.hide()

        self.flowLayout.addWidget(camera_widget)
        self.watch_trainer.add_camera(camera=camera)

    def selectCameraSource(self, camera, select_cam_btn, unselect_cam_btn):
        self.current_selected_source = camera

        if self.current_selected_source.image_processor.get_ptz_ready() == "ready":
            self.assign_network_ptz_btn.hide()
            self.unassign_network_ptz_btn.show()
        elif self.current_selected_source.image_processor.get_ptz_ready() == "not ready":
            self.assign_network_ptz_btn.show()
            self.unassign_network_ptz_btn.hide()
        else:
            self.assign_network_ptz_btn.hide()
            self.unassign_network_ptz_btn.hide()

        select_cam_btn.hide()
        unselect_cam_btn.show()

        self.select_face_dropdown.setEnabled(True)
        # Path for face image database

        if self.current_selected_source.image_processor.get_face() is None:
            self.select_face_dropdown.setCurrentText('')
            self.enable_track.setEnabled(False)
        else:
            self.select_face_dropdown.setCurrentText(self.current_selected_source.image_processor.get_face())
            self.enable_track.setEnabled(True)

        if self.current_selected_source.image_processor.is_track_enabled():
            self.enable_track.setChecked(True)
        else:
            self.enable_track.setChecked(False)

    def unselectCameraSource(self, select_cam_btn, unselect_cam_btn):
        self.current_selected_source = None
        self.select_face_dropdown.setCurrentText('')
        self.select_face_dropdown.setEnabled(False)
        self.enable_track.setChecked(False)
        self.enable_track.setEnabled(False)
        self.assign_network_ptz_btn.hide()
        self.unassign_network_ptz_btn.hide()
        unselect_cam_btn.hide()
        select_cam_btn.show()

    def deleteCameraSource(self, source, ndi_source, menu_item, camera, camera_widget):
        """Remove NDI/Serial camera source from camera grid"""
        menu_item.disconnect()
        if source == -1:
            # Remove NDI source widget
            menu_item.triggered.connect(lambda: self.addCamera(source=-1, ndi_source=ndi_source, menu_item=menu_item))
        else:
            # Remove Serial source widget
            menu_item.triggered.connect(lambda: self.addCamera(source=source, ndi_source=None, menu_item=menu_item))
            self.serial_widget_list.remove(camera)

        self.watch_trainer.remove_camera(camera=camera)
        camera.close()
        camera_widget.close()
        self.flowLayout.removeWidget(camera_widget)

    def translateUi(self, AutoPTZ):
        """Translate Menu, Buttons, and Labels through localization"""
        _translate = QtCore.QCoreApplication.translate
        AutoPTZ.setWindowTitle(_translate("AutoPTZ", "AutoPTZ"))
        self.enable_track.setText(_translate("AutoPTZ", "Enable Tracking"))
        self.select_face_label.setText(_translate("AutoPTZ", "Select Face"))
        self.assign_network_ptz_btn.setText(_translate("AutoPTZ", "Assign Network PTZ"))
        self.unassign_network_ptz_btn.setText(_translate("AutoPTZ", "Unassign Network PTZ"))
        self.formTabWidget.setTabText(self.formTabWidget.indexOf(self.selectedCamPage), _translate("AutoPTZ", "Auto"))
        self.select_camera_label.setText(_translate("AutoPTZ", "Select Camera"))
        self.down_right_btn.setText(_translate("AutoPTZ", "↘"))
        self.up_btn.setText(_translate("AutoPTZ", "↑"))
        self.up_left_btn.setText(_translate("AutoPTZ", "↖"))
        self.left_btn.setText(_translate("AutoPTZ", "←"))
        self.down_left_btn.setText(_translate("AutoPTZ", "↙"))
        self.up_right_btn.setText(_translate("AutoPTZ", "↗"))
        self.right_btn.setText(_translate("AutoPTZ", "→"))
        self.down_btn.setText(_translate("AutoPTZ", "↓"))
        self.home_btn.setText(_translate("AutoPTZ", "⌂"))
        self.zoom_in_btn.setText(_translate("AutoPTZ", "Zoom +"))
        self.zoom_out_btn.setText(_translate("AutoPTZ", "Zoom -"))
        self.focus_plus_btn.setText(_translate("AutoPTZ", "Focus +"))
        self.focus_minus_btn.setText(_translate("AutoPTZ", "Focus -"))
        self.menu_btn.setText(_translate("AutoPTZ", "Menu"))
        self.reset_btn.setText(_translate("AutoPTZ", "Reset"))
        self.assign_visca_ptz_btn.setText(_translate("AutoPTZ", "Assign VISCA PTZ"))
        self.unassign_visca_ptz_btn.setText(_translate("AutoPTZ", "Unassign VISCA PTZ"))
        self.formTabWidget.setTabText(self.formTabWidget.indexOf(self.manualControlPage),
                                      _translate("AutoPTZ", "Manual"))
        self.menuFile.setTitle(_translate("AutoPTZ", "File"))
        self.menuSource.setTitle(_translate("AutoPTZ", "Sources"))
        self.menuFacial_Recognition.setTitle(_translate("AutoPTZ", "Facial Recognition"))
        self.menuHelp.setTitle(_translate("AutoPTZ", "Help"))
        self.actionOpen.setText(_translate("AutoPTZ", "Open"))
        self.actionSave.setText(_translate("AutoPTZ", "Save"))
        self.actionSave_as.setText(_translate("AutoPTZ", "Save As"))
        self.actionClose.setText(_translate("AutoPTZ", "Close"))
        self.actionAdd_IP.setText(_translate("AutoPTZ", "Add IP"))
        self.menuAdd_NDI.setTitle(_translate("AutoPTZ", "Add NDI"))
        self.menuAdd_Hardware.setTitle(_translate("AutoPTZ", "Add Hardware"))
        self.actionEdit.setText(_translate("AutoPTZ", "Edit Setup"))
        self.actionContact.setText(_translate("AutoPTZ", "Contact"))
        self.actionAbout.setText(_translate("AutoPTZ", "About"))
        self.actionAdd_Face.setText(_translate("AutoPTZ", "Add Face"))
        self.actionTrain_Model.setText(_translate("AutoPTZ", "Retrain Model"))
        self.actionRemove_Face.setText(_translate("AutoPTZ", "Remove Face"))
        self.actionReset_Database.setText(_translate("AutoPTZ", "Reset Database"))
