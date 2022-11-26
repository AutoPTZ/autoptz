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
from libraries.move_visca_ptz import ViscaPTZ
from logic.camera_search.get_serial_cameras import COMPorts
from views.functions.show_dialogs_ui import ShowDialog
from views.functions.assign_network_ptz_ui import AssignNetworkPTZDlg
from views.functions.assign_visca_ptz_ui import AssignViscaPTZDlg
from views.homepage.flow_layout import FlowLayout
from shared.watch_trainer_directory import WatchTrainer
from views.widgets.camera_widget import CameraWidget


class AutoPTZ_MainWindow(QMainWindow):
    """
    Configures and Handles the AutoPTZ MainWindow UI
    """

    def __init__(self, *args, **kwargs):

        # setting up the UI and QT Threading
        super(AutoPTZ_MainWindow, self).__init__(*args, **kwargs)
        self.threadpool = QtCore.QThreadPool()
        self.threadpool.maxThreadCount()
        self.lock = Lock()

        # setting up main window
        self.setObjectName("AutoPTZ")
        self.resize(200, 450)
        self.setAutoFillBackground(False)
        self.setTabShape(QtWidgets.QTabWidget.TabShape.Rounded)
        self.setDockNestingEnabled(False)

        # base window widget
        self.central_widget = QtWidgets.QWidget(self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
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
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(self.formTabWidget.sizePolicy().hasHeightForWidth())
        self.formTabWidget.setSizePolicy(size_policy)
        self.formTabWidget.setObjectName("formTabWidget")

        # auto tab menu
        self.selectedCamPage = QtWidgets.QWidget(self)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(self.selectedCamPage.sizePolicy().hasHeightForWidth())
        self.selectedCamPage.setSizePolicy(size_policy)
        self.selectedCamPage.setMinimumSize(QtCore.QSize(163, 0))
        self.selectedCamPage.setMaximumSize(QtCore.QSize(16777215, 428))
        self.selectedCamPage.setObjectName("selectedCamPage")
        self.formLayout = QtWidgets.QFormLayout(self.selectedCamPage)
        self.formLayout.setLabelAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.formLayout.setFormAlignment(
            QtCore.Qt.AlignmentFlag.AlignLeading | QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.formLayout.setObjectName("formLayout")
        self.select_face_dropdown = QtWidgets.QComboBox(self.selectedCamPage)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy.setHeightForWidth(self.select_face_dropdown.sizePolicy().hasHeightForWidth())
        self.select_face_dropdown.setSizePolicy(size_policy)
        self.select_face_dropdown.setObjectName("select_face_dropdown")
        self.select_face_dropdown.setEnabled(False)
        self.select_face_dropdown.currentTextChanged.connect(self.selected_face_change)
        self.select_face_dropdown.addItem('')
        if os.path.isdir(constants.IMAGE_PATH):
            for folder in os.listdir(constants.IMAGE_PATH):
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

        self.formLayout.setWidget(2, QtWidgets.QFormLayout.ItemRole.SpanningRole, self.select_face_dropdown)
        self.enable_track = QtWidgets.QCheckBox(self.selectedCamPage)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy.setHeightForWidth(self.enable_track.sizePolicy().hasHeightForWidth())
        self.enable_track.setSizePolicy(size_policy)
        self.enable_track.setChecked(False)
        self.enable_track.setEnabled(False)
        self.enable_track.setAutoRepeat(False)
        self.enable_track.setAutoExclusive(False)
        self.enable_track.stateChanged.connect(self.enable_track_change)
        self.enable_track.setObjectName("enable_track")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.ItemRole.LabelRole, self.enable_track)
        self.select_face_label = QtWidgets.QLabel(self.selectedCamPage)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(self.select_face_label.sizePolicy().hasHeightForWidth())
        self.select_face_label.setSizePolicy(size_policy)
        self.select_face_label.setObjectName("select_face_label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.ItemRole.LabelRole, self.select_face_label)
        self.formTabWidget.addTab(self.selectedCamPage, "")

        # manual control tab menu
        self.manualControlPage = QtWidgets.QWidget()
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        size_policy.setHeightForWidth(self.manualControlPage.sizePolicy().hasHeightForWidth())
        self.manualControlPage.setSizePolicy(size_policy)
        self.manualControlPage.setMinimumSize(QtCore.QSize(163, 0))
        self.manualControlPage.setMaximumSize(QtCore.QSize(16777215, 428))
        self.manualControlPage.setObjectName("manualControlPage")
        self.select_camera_label = QtWidgets.QLabel(self.manualControlPage)
        self.select_camera_label.setGeometry(QtCore.QRect(10, 30, 101, 21))
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Maximum,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(self.select_camera_label.sizePolicy().hasHeightForWidth())
        self.select_camera_label.setSizePolicy(size_policy)
        self.select_camera_label.setObjectName("select_camera_label")
        self.select_camera_dropdown = QtWidgets.QComboBox(self.manualControlPage)
        self.select_camera_dropdown.setGeometry(QtCore.QRect(9, 51, 151, 26))
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.MinimumExpanding,
                                            QtWidgets.QSizePolicy.Policy.Fixed)
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
        # self.controller_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.controller_layout.setContentsMargins(0, 0, 0, 0)
        self.controller_layout.setObjectName("controllerLayout")
        self.down_right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.down_right_btn.sizePolicy().hasHeightForWidth())
        self.down_right_btn.setSizePolicy(size_policy)
        self.down_right_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_right_btn.setFlat(False)
        self.down_right_btn.setObjectName("down_right_btn")
        self.controller_layout.addWidget(self.down_right_btn, 2, 2, 1, 1)
        self.up_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.up_btn.sizePolicy().hasHeightForWidth())
        self.up_btn.setSizePolicy(size_policy)
        self.up_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_btn.setObjectName("up_btn")
        self.controller_layout.addWidget(self.up_btn, 0, 1, 1, 1)
        self.up_left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.up_left_btn.sizePolicy().hasHeightForWidth())
        self.up_left_btn.setSizePolicy(size_policy)
        self.up_left_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_left_btn.setObjectName("up_left_btn")
        self.controller_layout.addWidget(self.up_left_btn, 0, 0, 1, 1)
        self.left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.left_btn.sizePolicy().hasHeightForWidth())
        self.left_btn.setSizePolicy(size_policy)
        self.left_btn.setIconSize(QtCore.QSize(10, 10))
        self.left_btn.setObjectName("left_btn")
        self.controller_layout.addWidget(self.left_btn, 1, 0, 1, 1)
        self.down_left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.down_left_btn.sizePolicy().hasHeightForWidth())
        self.down_left_btn.setSizePolicy(size_policy)
        self.down_left_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_left_btn.setObjectName("down_left_btn")
        self.controller_layout.addWidget(self.down_left_btn, 2, 0, 1, 1)
        self.up_right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.up_right_btn.sizePolicy().hasHeightForWidth())
        self.up_right_btn.setSizePolicy(size_policy)
        self.up_right_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_right_btn.setObjectName("up_right_btn")
        self.controller_layout.addWidget(self.up_right_btn, 0, 2, 1, 1)
        self.right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.right_btn.sizePolicy().hasHeightForWidth())
        self.right_btn.setSizePolicy(size_policy)
        self.right_btn.setIconSize(QtCore.QSize(10, 10))
        self.right_btn.setObjectName("right_btn")
        self.controller_layout.addWidget(self.right_btn, 1, 2, 1, 1)
        self.down_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
        size_policy.setHeightForWidth(self.down_btn.sizePolicy().hasHeightForWidth())
        self.down_btn.setSizePolicy(size_policy)
        self.down_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_btn.setObjectName("down_btn")
        self.controller_layout.addWidget(self.down_btn, 2, 1, 1, 1)
        self.home_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Maximum)
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
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHeightForWidth(self.zoom_in_btn.sizePolicy().hasHeightForWidth())
        self.zoom_in_btn.setSizePolicy(size_policy)
        self.zoom_in_btn.setObjectName("zoom_in_btn")
        self.zoom_layout.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
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
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.focus_plus_btn.sizePolicy().hasHeightForWidth())
        self.focus_plus_btn.setSizePolicy(size_policy)
        self.focus_plus_btn.setObjectName("focus_plus_btn")
        self.focus_layout.addWidget(self.focus_plus_btn)
        self.focus_minus_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
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
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
        size_policy.setHorizontalStretch(0)
        size_policy.setVerticalStretch(0)
        size_policy.setHeightForWidth(self.menu_btn.sizePolicy().hasHeightForWidth())
        self.menu_btn.setSizePolicy(size_policy)
        self.menu_btn.setObjectName("menu_btn")
        self.menu_layout.addWidget(self.menu_btn)
        self.reset_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                                            QtWidgets.QSizePolicy.Policy.Preferred)
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
        self.flowLayout = FlowLayout()
        self.gridLayout.addLayout(self.flowLayout, 0, 1, 1, 1)

        # handling camera window sizing
        self.screen_width = self.screen().availableGeometry().width()
        self.screen_height = self.screen().availableGeometry().height()

        # Create Dialog Object
        self.dialogs = ShowDialog()

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
        self.actionOpen = QtWidgets.QWidgetAction(self)
        self.actionOpen.setObjectName("actionOpen")
        self.actionSave = QtWidgets.QWidgetAction(self)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtWidgets.QWidgetAction(self)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionClose = QtWidgets.QWidgetAction(self)
        self.actionClose.setObjectName("actionClose")
        self.actionAdd_IP = QtWidgets.QWidgetAction(self)
        self.actionAdd_IP.setObjectName("actionAdd_IP")
        self.menuAdd_NDI = QtWidgets.QMenu(self)
        self.menuAdd_NDI.setObjectName("menuAdd_NDI")
        self.menuAdd_Hardware = QtWidgets.QMenu(self)
        self.menuAdd_Hardware.setObjectName("menuAdd_Hardware")
        self.actionEdit = QtWidgets.QWidgetAction(self)
        self.actionEdit.setObjectName("actionEdit")
        self.actionContact = QtWidgets.QWidgetAction(self)
        self.actionContact.setObjectName("actionContact")
        self.actionAbout = QtWidgets.QWidgetAction(self)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAdd_Face = QtWidgets.QWidgetAction(self)
        self.actionAdd_Face.setObjectName("actionAdd_Face")
        self.actionAdd_Face.triggered.connect(partial(self.dialogs.add_face, self.update_face_dropdown))
        self.actionTrain_Model = QtWidgets.QWidgetAction(self)
        self.actionTrain_Model.setObjectName("actionTrain_Model")
        self.actionTrain_Model.triggered.connect(partial(self.dialogs.retrain_face))
        self.actionRemove_Face = QtWidgets.QWidgetAction(self)
        self.actionRemove_Face.setObjectName("actionRemove_Face")
        self.actionRemove_Face.triggered.connect(partial(self.dialogs.remove_face, self.update_face_dropdown))
        self.actionReset_Database = QtWidgets.QWidgetAction(self)
        self.actionReset_Database.setObjectName("actionReset_Database")
        self.actionReset_Database.triggered.connect(partial(self.dialogs.reset_database, self.update_face_dropdown))
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

        self.findNDISources()
        self.findHardwareSources()
        if os.path.exists(constants.TRAINER_PATH) is False:
            os.mkdir(constants.TRAINER_PATH)
        self.watch_trainer = WatchTrainer()
        observer = watchdog.observers.Observer()
        observer.schedule(self.watch_trainer, path=constants.TRAINER_PATH, recursive=True)
        observer.start()
        self.translateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def findHardwareSources(self):
        """Adds camera sources to the Hardware source list"""
        available_cameras = QMediaDevices.videoInputs()

        for index, cam in enumerate(available_cameras):
            menu_item = QtWidgets.QWidgetAction(self)
            menu_item.setText(cam.description())
            menu_item.setCheckable(True)
            menu_item.triggered.connect(self.create_lambda(src=index, menu_item=menu_item, isNDI=False))
            self.menuAdd_Hardware.addAction(menu_item)

    def findNDISources(self):
        """Adds NDI sources to the NDI source list"""
        source_list = get_ndi_sources()
        for index, cam in enumerate(source_list):
            menu_item = QtWidgets.QWidgetAction(self)
            menu_item.setText(cam.ndi_name)
            menu_item.setCheckable(True)
            menu_item.triggered.connect(self.create_lambda(src=cam, menu_item=menu_item, isNDI=True))
            self.menuAdd_NDI.addAction(menu_item)

    def create_lambda(self, src, menu_item, isNDI):
        """
        Fixes MenuItem late assignment for Camera Sources by returning lambda statement
        :param src:
        :param menu_item:
        :param isNDI:
        :return:
        """
        return lambda: self.addCameraWidget(source=src, menu_item=menu_item, isNDI=isNDI)

    def addCameraWidget(self, source, menu_item, isNDI=False):
        """Add NDI/Serial camera source from the menu to the FlowLayout"""
        camera_widget = CameraWidget(source=source, width=self.screen_width // 3, height=self.screen_height // 3,
                                     isNDI=isNDI, lock=self.lock)
        camera_widget.change_selection_signal.connect(self.updateElements)
        menu_item.triggered.disconnect()
        menu_item.triggered.connect(
            lambda index=source, item=menu_item: self.deleteCameraWidget(source=index, menu_item=item,
                                                                         camera_widget=camera_widget))
        self.watch_trainer.add_camera(camera_widget=camera_widget)
        self.flowLayout.addWidget(camera_widget)
        camera_widget.show()

    def deleteCameraWidget(self, source, menu_item, camera_widget):
        """Remove NDI/Serial camera source from camera FlowLayout"""
        menu_item.triggered.disconnect()
        menu_item.triggered.connect(
            lambda index=source, item=menu_item: self.addCameraWidget(source=index, menu_item=item))
        self.watch_trainer.remove_camera(camera_widget=camera_widget)
        if constants.CURRENT_ACTIVE_CAM_WIDGET == camera_widget:
            constants.CURRENT_ACTIVE_CAM_WIDGET = None
            self.updateElements()
        camera_widget.stop()
        camera_widget.deleteLater()

    def updateElements(self):
        """
        Update UI elements like FaceDropDownMenu and Enable Track Checkbox when a CameraWidget is activated/deactivated
        """
        if constants.CURRENT_ACTIVE_CAM_WIDGET is None:
            print(f"No Camera Source is active")
            self.select_face_dropdown.setEnabled(False)
            self.select_face_dropdown.setCurrentText('')
            self.enable_track.setEnabled(False)
            self.enable_track.setChecked(False)
            self.assign_network_ptz_btn.hide()
            self.assign_network_ptz_btn.hide()
        else:
            print(f"{constants.CURRENT_ACTIVE_CAM_WIDGET.objectName()} is active")
            self.select_face_dropdown.setEnabled(True)
            if constants.CURRENT_ACTIVE_CAM_WIDGET.processor_thread is not None:
                print("Processor Thread is running")
                if constants.CURRENT_ACTIVE_CAM_WIDGET.get_tracked_name() is None:
                    print("no tracked name")
                    self.select_face_dropdown.setCurrentText('')
                    self.enable_track.blockSignals(True)
                    self.enable_track.setEnabled(False)
                    self.enable_track.setChecked(False)
                    self.enable_track.blockSignals(False)
                else:
                    self.select_face_dropdown.setCurrentText(constants.CURRENT_ACTIVE_CAM_WIDGET.get_tracked_name())
                    self.enable_track.blockSignals(True)
                    self.enable_track.setChecked(True)
                    self.enable_track.blockSignals(False)
                    if constants.CURRENT_ACTIVE_CAM_WIDGET.get_tracking() is False:
                        self.enable_track.blockSignals(True)
                        self.enable_track.setChecked(False)
                        self.enable_track.blockSignals(False)
                        print(
                            f"a tracked name is {constants.CURRENT_ACTIVE_CAM_WIDGET.get_tracked_name()} but tracking is disabled")
                    else:
                        self.enable_track.blockSignals(True)
                        self.enable_track.setChecked(True)
                        self.enable_track.blockSignals(False)
                        print(
                            f"a tracked name is {constants.CURRENT_ACTIVE_CAM_WIDGET.get_tracked_name()} and tracking is enabled")
            else:
                print("Processor Thread is not running")
                self.select_face_dropdown.setEnabled(True)
                self.select_face_dropdown.setCurrentText('')
                self.enable_track.setEnabled(False)
                self.enable_track.setChecked(False)
            if constants.CURRENT_ACTIVE_CAM_WIDGET.isNDI and constants.CURRENT_ACTIVE_CAM_WIDGET.ptz_control_thread is None:
                self.unassign_network_ptz_btn.hide()
                self.assign_network_ptz_btn.show()
            elif constants.CURRENT_ACTIVE_CAM_WIDGET.isNDI and constants.CURRENT_ACTIVE_CAM_WIDGET.ptz_control_thread is not None:
                self.unassign_network_ptz_btn.show()
                self.assign_network_ptz_btn.hide()
            else:
                self.assign_network_ptz_btn.hide()
                self.assign_network_ptz_btn.hide()

    def selected_face_change(self):
        """
        Update Current Active CameraWidget's Tracked Name and UI
        """
        if constants.CURRENT_ACTIVE_CAM_WIDGET is not None:
            if self.select_face_dropdown.currentText() == '':
                constants.CURRENT_ACTIVE_CAM_WIDGET.set_tracked_name(None)
                self.enable_track.setEnabled(False)
                self.enable_track.setChecked(False)
            else:
                constants.CURRENT_ACTIVE_CAM_WIDGET.set_tracked_name(self.select_face_dropdown.currentText())
                self.enable_track.setEnabled(True)

    def enable_track_change(self):
        """
        Update Current Active CameraWidget's Enable/Disable Tracking and UI
        """
        if constants.CURRENT_ACTIVE_CAM_WIDGET is not None:
            print(f"setting track button for {self.enable_track.isChecked()}")
            constants.CURRENT_ACTIVE_CAM_WIDGET.set_tracking()

    def update_face_dropdown(self, event):
        """
        Update Face Dropdown List when faces are added or removed
        """
        current_text_temp = self.select_face_dropdown.currentText()
        self.select_face_dropdown.clear()
        self.select_face_dropdown.addItem('')
        if os.path.exists(constants.IMAGE_PATH):
            for folder in os.listdir(constants.IMAGE_PATH):
                self.select_face_dropdown.addItem(folder)
            if self.select_face_dropdown.findText(current_text_temp) != -1:
                self.select_face_dropdown.setCurrentText(current_text_temp)

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
            self.up_left_btn.clicked.disconnect()
            self.up_btn.clicked.disconnect()
            self.up_right_btn.clicked.disconnect()
            self.left_btn.clicked.disconnect()
            self.right_btn.clicked.disconnect()
            self.down_left_btn.clicked.disconnect()
            self.down_btn.clicked.disconnect()
            self.down_right_btn.clicked.disconnect()
            self.home_btn.clicked.disconnect()
            self.zoom_in_btn.clicked.disconnect()
            self.zoom_out_btn.clicked.disconnect()
            self.menu_btn.clicked.disconnect()
            self.reset_btn.clicked.disconnect()

        # shows button depending on if device has already been assigned to a camera source
        try:
            if device in self.assigned_ptz_camera:
                self.unassign_visca_ptz_btn.show()
            else:
                self.assign_visca_ptz_btn.show()
        except Exception as e:
            print(e)
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
        if constants.CURRENT_ACTIVE_CAM_WIDGET is None:
            print("Need to select or add a camera")
        else:
            dlg = AssignNetworkPTZDlg(self, camera=constants.CURRENT_ACTIVE_CAM_WIDGET)
            dlg.closeEvent = self.refreshNetworkBtn
            dlg.exec()

    def unassign_network_ptz(self):
        """Allow User to Unassign current Network PTZ device from Camera Source"""
        constants.CURRENT_ACTIVE_CAM_WIDGET.set_ptz(control=None)
        self.unassign_network_ptz_btn.hide()
        self.assign_network_ptz_btn.show()

    def refreshViscaBtn(self, event):
        """Check is VISCA PTZ is assigned and change assignment button if so"""
        if self.select_camera_dropdown.currentText() in self.assigned_ptz_camera:
            self.unassign_visca_ptz_btn.show()
            self.assign_visca_ptz_btn.hide()
        else:
            self.assign_visca_ptz_btn.show()
            self.unassign_visca_ptz_btn.hide()

    def refreshNetworkBtn(self, event):
        """Check is Network PTZ is assigned and change assignment button if so"""
        if constants.CURRENT_ACTIVE_CAM_WIDGET is not None:
            self.unassign_network_ptz_btn.show()
            self.assign_network_ptz_btn.hide()
        else:
            self.assign_network_ptz_btn.show()
            self.unassign_network_ptz_btn.hide()

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
