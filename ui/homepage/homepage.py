import os

import cv2
from PyQt5 import QtCore, QtWidgets
import watchdog.events
import watchdog.observers

from logic.facial_tracking.add_face import AddFaceDlg
from logic.facial_tracking.remove_face import RemoveFaceDlg
from logic.facial_tracking.reset_database import ResetDatabaseDlg
from logic.facial_tracking.train_face import Trainer
from ui.homepage.move_visca_ptz import ViscaPTZ
from ui.homepage.assign_ptz_ui import AssignPTZDlg
from ui.homepage.flow_layout import FlowLayout
from logic.camera_search.get_serial_cameras import COMPorts
from logic.camera_search.search_ndi import get_ndi_sources
from ui.shared.message_prompts import show_info_messagebox
from ui.shared.watch_trainer_directory import WatchTrainer
from ui.widgets.camera_widget import CameraWidget
from ui.widgets.ndi_cam_widget import NDICameraWidget


class Ui_AutoPTZ(object):
    def __init__(self):
        self.image_path = None
        self.watch_trainer = None
        self.current_manual_device = None
        self.current_selected_source = None
        self.actionReset_Database = None
        self.actionRemove_Face = None
        self.actionTrain_Model = None
        self.actionAdd_Face = None
        self.menuFacial_Recognition = None
        self.actionAbout = None
        self.actionContact = None
        self.actionEdit = None
        self.menuAdd_Hardware = None
        self.menuAdd_NDI = None
        self.actionAdd_IP = None
        self.actionClose = None
        self.actionSave_as = None
        self.actionSave = None
        self.actionOpen = None
        self.statusbar = None
        self.menuHelp = None
        self.menuSource = None
        self.menuFile = None
        self.menubar = None
        self.screen_height = None
        self.screen_width = None
        self.flowLayout = None
        self.shown_cameras = None
        self.unassign_ptz_btn = None
        self.assign_ptz_btn = None
        self.reset_btn = None
        self.menu_btn = None
        self.menu_layout = None
        self.horizontalLayoutWidget_3 = None
        self.focus_minus_btn = None
        self.focus_plus_btn = None
        self.focus_layout = None
        self.horizontalLayoutWidget_2 = None
        self.zoom_out_btn = None
        self.zoom_in_btn = None
        self.zoom_layout = None
        self.horizontalLayoutWidget = None
        self.home_btn = None
        self.down_btn = None
        self.right_btn = None
        self.up_right_btn = None
        self.down_left_btn = None
        self.left_btn = None
        self.up_left_btn = None
        self.up_btn = None
        self.down_right_btn = None
        self.controller_layout = None
        self.gridLayoutWidget = None
        self.select_camera_dropdown = None
        self.select_camera_label = None
        self.manualControlPage = None
        self.select_face_label = None
        self.select_face_dropdown = None
        self.enable_track = None
        self.formLayout = None
        self.selectedCamPage = None
        self.formTabWidget = None
        self.gridLayout = None
        self.central_widget = None
        self.assigned_ptz_camera = None
        self.serial_widget_list = None

    def setupUi(self, AutoPTZ):
        # setting up home window
        AutoPTZ.setObjectName("AutoPTZ")
        AutoPTZ.resize(200, 450)
        AutoPTZ.setAutoFillBackground(False)
        AutoPTZ.setTabShape(QtWidgets.QTabWidget.Rounded)
        AutoPTZ.setDockNestingEnabled(False)

        # base window widget
        self.central_widget = QtWidgets.QWidget(AutoPTZ)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.central_widget.sizePolicy().hasHeightForWidth())
        self.central_widget.setSizePolicy(size_policy)
        self.central_widget.setObjectName("central_widget")
        self.gridLayout = QtWidgets.QGridLayout(self.central_widget)
        self.gridLayout.setObjectName("gridLayout")

        # left tab menus
        self.formTabWidget = QtWidgets.QTabWidget(self.central_widget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.formTabWidget.sizePolicy().hasHeightForWidth())
        self.formTabWidget.setSizePolicy(size_policy)
        self.formTabWidget.setObjectName("formTabWidget")

        # auto tab menu
        self.selectedCamPage = QtWidgets.QWidget()
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        size_policy.setHeightForWidth(self.selectedCamPage.sizePolicy().hasHeightForWidth())
        self.selectedCamPage.setSizePolicy(size_policy)
        self.selectedCamPage.setMinimumSize(QtCore.QSize(150, 0))
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
        self.image_path = '../logic/facial_tracking/images/'
        self.select_face_dropdown.currentTextChanged.connect(self.selected_face_change)

        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.select_face_dropdown)
        self.enable_track = QtWidgets.QCheckBox(self.selectedCamPage)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        size_policy.setHeightForWidth(self.enable_track.sizePolicy().hasHeightForWidth())
        self.enable_track.setSizePolicy(size_policy)
        self.enable_track.setChecked(False)
        self.enable_track.setAutoRepeat(False)
        self.enable_track.setAutoExclusive(False)
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
        self.manualControlPage.setMinimumSize(QtCore.QSize(162, 0))
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
        self.assign_ptz_btn = QtWidgets.QPushButton(self.manualControlPage)
        self.assign_ptz_btn.setGeometry(QtCore.QRect(10, 380, 141, 32))
        self.assign_ptz_btn.setObjectName("assign_ptz_btn")
        self.assign_ptz_btn.hide()
        self.unassign_ptz_btn = QtWidgets.QPushButton(self.manualControlPage)
        self.unassign_ptz_btn.setGeometry(QtCore.QRect(10, 380, 141, 32))
        self.unassign_ptz_btn.setObjectName("unassign_ptz_btn")
        self.unassign_ptz_btn.hide()
        self.assign_ptz_btn.clicked.connect(self.assign_ptz_dialog)
        self.unassign_ptz_btn.clicked.connect(self.unassign_ptz)
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
        AutoPTZ.setCentralWidget(self.central_widget)
        self.menubar = QtWidgets.QMenuBar(AutoPTZ)
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
        AutoPTZ.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AutoPTZ)
        self.statusbar.setObjectName("statusbar")
        AutoPTZ.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(AutoPTZ)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QAction(AutoPTZ)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtWidgets.QAction(AutoPTZ)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionClose = QtWidgets.QAction(AutoPTZ)
        self.actionClose.setObjectName("actionClose")
        self.actionAdd_IP = QtWidgets.QAction(AutoPTZ)
        self.actionAdd_IP.setObjectName("actionAdd_IP")
        self.menuAdd_NDI = QtWidgets.QMenu(AutoPTZ)
        self.menuAdd_NDI.setObjectName("menuAdd_NDI")
        self.menuAdd_Hardware = QtWidgets.QMenu(AutoPTZ)
        self.menuAdd_Hardware.setObjectName("menuAdd_Hardware")
        self.actionEdit = QtWidgets.QAction(AutoPTZ)
        self.actionEdit.setObjectName("actionEdit")
        self.actionContact = QtWidgets.QAction(AutoPTZ)
        self.actionContact.setObjectName("actionContact")
        self.actionAbout = QtWidgets.QAction(AutoPTZ)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAdd_Face = QtWidgets.QAction(AutoPTZ)
        self.actionAdd_Face.setObjectName("actionAdd_Face")
        self.actionAdd_Face.triggered.connect(self.add_face)
        self.actionTrain_Model = QtWidgets.QAction(AutoPTZ)
        self.actionTrain_Model.setObjectName("actionTrain_Model")
        self.actionTrain_Model.triggered.connect(self.retrain_face)
        self.actionRemove_Face = QtWidgets.QAction(AutoPTZ)
        self.actionRemove_Face.setObjectName("actionRemove_Face")
        self.actionRemove_Face.triggered.connect(self.remove_face)
        self.actionReset_Database = QtWidgets.QAction(AutoPTZ)
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
        self.watch_trainer = WatchTrainer()
        observer = watchdog.observers.Observer()
        observer.schedule(self.watch_trainer, path="../logic/facial_tracking/trainer/", recursive=True)
        observer.start()
        self.translateUi(AutoPTZ)
        QtCore.QMetaObject.connectSlotsByName(AutoPTZ)

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
                self.unassign_ptz_btn.show()
            else:
                self.assign_ptz_btn.show()
        except:
            self.assign_ptz_btn.hide()
            self.unassign_ptz_btn.hide()

    def assign_ptz_dialog(self):
        """Launch the Assign PTZ to Camera Source dialog."""
        if not self.serial_widget_list or self.select_camera_dropdown.currentText() == "":
            print("Need to select or add a camera")
        else:
            dlg = AssignPTZDlg(self, camera_list=self.serial_widget_list, assigned_list=self.assigned_ptz_camera,
                               ptz_id=self.select_camera_dropdown.currentText())
            dlg.closeEvent = self.refreshBtnOnClose
            dlg.exec()

    def unassign_ptz(self):
        """Allow User to Unassign current VISCA PTZ device from Camera Source"""
        index = self.assigned_ptz_camera.index(self.select_camera_dropdown.currentText())

        camera = self.assigned_ptz_camera[index + 1]
        camera.set_tracker(None)
        self.assigned_ptz_camera.remove(camera)
        self.assigned_ptz_camera.remove(self.select_camera_dropdown.currentText())

        self.unassign_ptz_btn.hide()
        self.assign_ptz_btn.show()

    def add_face(self):
        """Launch the Add Face dialog based on the currently selected camera."""
        if self.flowLayout.count() == 0 or self.current_selected_source is None:
            show_info_messagebox("Please add and select a camera.")
        else:
            print("Opening Face Dialog")
            dlg = AddFaceDlg(self, camera=self.current_selected_source)
            dlg.exec()

    @staticmethod
    def retrain_face():
        if not os.path.isdir('../logic/facial_tracking/images/'):
            show_info_messagebox("No Faces to train.")
        else:
            Trainer().train_face(True)

    def remove_face(self):
        """Launch the Remove Face dialog based on the currently selected camera."""
        if not os.path.isdir('../logic/facial_tracking/images/'):
            show_info_messagebox("No Faces to remove.")
        else:
            print("Opening Face Dialog")
            dlg = RemoveFaceDlg(self)
            dlg.exec()

    def reset_database(self):
        """Launch the Remove Face dialog based on the currently selected camera."""
        print("Opening Face Dialog")
        dlg = ResetDatabaseDlg(self)
        dlg.exec()

    def refreshBtnOnClose(self, event):
        """Check is VISCA PTZ is assigned and change assignment button if so"""
        if self.select_camera_dropdown.currentText() in self.assigned_ptz_camera:
            self.unassign_ptz_btn.show()
        else:
            self.assign_ptz_btn.show()

    def check_all_faces(self):
        self.select_face_dropdown.clear()
        self.select_face_dropdown.addItem("")
        # Path for face image database
        if os.path.exists(self.image_path):
            for folder in os.listdir(self.image_path):
                self.select_face_dropdown.addItem(folder)

    def selected_face_change(self):
        try:
            self.current_selected_source.changeFace(self.select_face_dropdown.currentText())
        except:
            self.current_selected_source.changeFace('')

    def getPhysicalSourcesList(self):
        """Test ports 0-6 and adds all camera sources to the physical source list"""
        non_working_ports = []
        dev_port = 0
        working_ports = []
        available_ports = []
        while len(non_working_ports) < 6:
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    working_ports.append(dev_port)
                    self.addPhysicalSource("Camera %s (%s x %s)" % (dev_port + 1, w, h), dev_port)
                else:
                    available_ports.append(dev_port)
            dev_port += 1
        return

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
        select_cam_btn.hide()
        unselect_cam_btn.show()
        self.check_all_faces()

    def unselectCameraSource(self, select_cam_btn, unselect_cam_btn):
        self.current_selected_source = None
        unselect_cam_btn.hide()
        select_cam_btn.show()
        self.select_face_dropdown.clear()


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
        self.assign_ptz_btn.setText(_translate("AutoPTZ", "Assign PTZ"))
        self.unassign_ptz_btn.setText(_translate("AutoPTZ", "Unassign PTZ"))
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
