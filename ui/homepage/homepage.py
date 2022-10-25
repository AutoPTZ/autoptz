import cv2
from PyQt5 import QtCore, QtWidgets

from ui.homepage.assign_ptz_ui import AssignCamDlg
from ui.homepage.flow_layout_test import FlowLayout
from visca import camera
import time
import datetime
import threading

from logic.camera_search.get_serial_cameras import COMPorts
from logic.camera_search.search_ndi import get_ndi_sources
from ui.widgets.camera_widget import CameraWidget
from ui.widgets.ndi_cam_widget import NDICameraWidget

camera_widget_list = []
assigned_ptz_camera = []

current_manual_device = None


class Ui_AutoPTZ(object):
    def setupUi(self, AutoPTZ):
        self.assigned_ptz_camera = assigned_ptz_camera
        AutoPTZ.setObjectName("AutoPTZ")
        AutoPTZ.resize(200, 450)
        AutoPTZ.setAutoFillBackground(False)
        AutoPTZ.setTabShape(QtWidgets.QTabWidget.Rounded)
        AutoPTZ.setDockNestingEnabled(False)
        self.centralwidget = QtWidgets.QWidget(AutoPTZ)

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.formTabWidget = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.formTabWidget.sizePolicy().hasHeightForWidth())
        self.formTabWidget.setSizePolicy(sizePolicy)
        self.formTabWidget.setObjectName("formTabWidget")
        self.selectedCamPage = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.selectedCamPage.sizePolicy().hasHeightForWidth())
        self.selectedCamPage.setSizePolicy(sizePolicy)
        self.selectedCamPage.setMinimumSize(QtCore.QSize(150, 0))
        self.selectedCamPage.setMaximumSize(QtCore.QSize(16777215, 428))
        self.selectedCamPage.setObjectName("selectedCamPage")
        self.formLayout = QtWidgets.QFormLayout(self.selectedCamPage)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setObjectName("formLayout")
        self.select_face_dropdown = QtWidgets.QComboBox(self.selectedCamPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_face_dropdown.sizePolicy().hasHeightForWidth())
        self.select_face_dropdown.setSizePolicy(sizePolicy)
        self.select_face_dropdown.setObjectName("select_face_dropdown")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.select_face_dropdown)
        self.enable_track = QtWidgets.QCheckBox(self.selectedCamPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.enable_track.sizePolicy().hasHeightForWidth())
        self.enable_track.setSizePolicy(sizePolicy)
        self.enable_track.setChecked(False)
        self.enable_track.setAutoRepeat(False)
        self.enable_track.setAutoExclusive(False)
        self.enable_track.setObjectName("enable_track")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.enable_track)
        self.select_face_label = QtWidgets.QLabel(self.selectedCamPage)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_face_label.sizePolicy().hasHeightForWidth())
        self.select_face_label.setSizePolicy(sizePolicy)
        self.select_face_label.setObjectName("select_face_label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.select_face_label)
        self.formTabWidget.addTab(self.selectedCamPage, "")
        self.manualControlPage = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.manualControlPage.sizePolicy().hasHeightForWidth())
        self.manualControlPage.setSizePolicy(sizePolicy)
        self.manualControlPage.setMinimumSize(QtCore.QSize(162, 0))
        self.manualControlPage.setMaximumSize(QtCore.QSize(16777215, 428))
        self.manualControlPage.setObjectName("manualControlPage")
        self.select_camera_label = QtWidgets.QLabel(self.manualControlPage)
        self.select_camera_label.setGeometry(QtCore.QRect(10, 30, 101, 21))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_camera_label.sizePolicy().hasHeightForWidth())
        self.select_camera_label.setSizePolicy(sizePolicy)
        self.select_camera_label.setObjectName("select_camera_label")
        self.select_camera_dropdown = QtWidgets.QComboBox(self.manualControlPage)
        self.select_camera_dropdown.setGeometry(QtCore.QRect(9, 51, 151, 26))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.select_camera_dropdown.sizePolicy().hasHeightForWidth())
        self.select_camera_dropdown.setSizePolicy(sizePolicy)
        self.select_camera_dropdown.setObjectName("select_camera_dropdown")

        self.select_camera_dropdown.addItem("")
        data_list = COMPorts.get_com_ports().data
        for port in data_list:
            if "USB" in port.description:
                print(port.device, port.description, data_list.index(port))
                self.select_camera_dropdown.addItem(port.device)

        self.select_camera_dropdown.currentTextChanged.connect(self.init_manual_control)

        self.gridLayoutWidget = QtWidgets.QWidget(self.manualControlPage)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 100, 162, 131))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.contollerLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.contollerLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.contollerLayout.setContentsMargins(0, 0, 0, 0)
        self.contollerLayout.setObjectName("contollerLayout")
        self.down_right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.down_right_btn.sizePolicy().hasHeightForWidth())
        self.down_right_btn.setSizePolicy(sizePolicy)
        self.down_right_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_right_btn.setFlat(False)
        self.down_right_btn.setObjectName("down_right_btn")
        self.contollerLayout.addWidget(self.down_right_btn, 2, 2, 1, 1)
        self.up_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.up_btn.sizePolicy().hasHeightForWidth())
        self.up_btn.setSizePolicy(sizePolicy)
        self.up_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_btn.setObjectName("up_btn")
        self.contollerLayout.addWidget(self.up_btn, 0, 1, 1, 1)
        self.up_left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.up_left_btn.sizePolicy().hasHeightForWidth())
        self.up_left_btn.setSizePolicy(sizePolicy)
        self.up_left_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_left_btn.setObjectName("up_left_btn")
        self.contollerLayout.addWidget(self.up_left_btn, 0, 0, 1, 1)
        self.left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_btn.sizePolicy().hasHeightForWidth())
        self.left_btn.setSizePolicy(sizePolicy)
        self.left_btn.setIconSize(QtCore.QSize(10, 10))
        self.left_btn.setObjectName("left_btn")
        self.contollerLayout.addWidget(self.left_btn, 1, 0, 1, 1)
        self.down_left_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.down_left_btn.sizePolicy().hasHeightForWidth())
        self.down_left_btn.setSizePolicy(sizePolicy)
        self.down_left_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_left_btn.setObjectName("down_left_btn")
        self.contollerLayout.addWidget(self.down_left_btn, 2, 0, 1, 1)
        self.up_right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.up_right_btn.sizePolicy().hasHeightForWidth())
        self.up_right_btn.setSizePolicy(sizePolicy)
        self.up_right_btn.setIconSize(QtCore.QSize(10, 10))
        self.up_right_btn.setObjectName("up_right_btn")
        self.contollerLayout.addWidget(self.up_right_btn, 0, 2, 1, 1)
        self.right_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_btn.sizePolicy().hasHeightForWidth())
        self.right_btn.setSizePolicy(sizePolicy)
        self.right_btn.setIconSize(QtCore.QSize(10, 10))
        self.right_btn.setObjectName("right_btn")
        self.contollerLayout.addWidget(self.right_btn, 1, 2, 1, 1)
        self.down_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.down_btn.sizePolicy().hasHeightForWidth())
        self.down_btn.setSizePolicy(sizePolicy)
        self.down_btn.setIconSize(QtCore.QSize(10, 10))
        self.down_btn.setObjectName("down_btn")
        self.contollerLayout.addWidget(self.down_btn, 2, 1, 1, 1)
        self.home_btn = QtWidgets.QPushButton(self.gridLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.home_btn.sizePolicy().hasHeightForWidth())
        self.home_btn.setSizePolicy(sizePolicy)
        self.home_btn.setIconSize(QtCore.QSize(10, 10))
        self.home_btn.setObjectName("home_btn")
        self.contollerLayout.addWidget(self.home_btn, 1, 1, 1, 1)
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.manualControlPage)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(0, 240, 161, 32))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.zoom_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.zoom_layout.setContentsMargins(0, 0, 0, 0)
        self.zoom_layout.setObjectName("zoom_layout")
        self.zoom_in_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.zoom_in_btn.sizePolicy().hasHeightForWidth())
        self.zoom_in_btn.setSizePolicy(sizePolicy)
        self.zoom_in_btn.setObjectName("zoom_in_btn")
        self.zoom_layout.addWidget(self.zoom_in_btn)
        self.zoom_out_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.zoom_out_btn.sizePolicy().hasHeightForWidth())
        self.zoom_out_btn.setSizePolicy(sizePolicy)
        self.zoom_out_btn.setObjectName("zoom_out_btn")
        self.zoom_layout.addWidget(self.zoom_out_btn)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.manualControlPage)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 280, 161, 32))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.focus_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.focus_layout.setContentsMargins(0, 0, 0, 0)
        self.focus_layout.setObjectName("focus_layout")
        self.focus_plus_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.focus_plus_btn.sizePolicy().hasHeightForWidth())
        self.focus_plus_btn.setSizePolicy(sizePolicy)
        self.focus_plus_btn.setObjectName("focus_plus_btn")
        self.focus_layout.addWidget(self.focus_plus_btn)
        self.focus_minus_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.focus_minus_btn.sizePolicy().hasHeightForWidth())
        self.focus_minus_btn.setSizePolicy(sizePolicy)
        self.focus_minus_btn.setObjectName("focus_minus_btn")
        self.focus_layout.addWidget(self.focus_minus_btn)
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.manualControlPage)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(0, 320, 161, 32))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.menu_layout = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.menu_layout.setContentsMargins(0, 0, 0, 0)
        self.menu_layout.setObjectName("menu_layout")
        self.menu_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.menu_btn.sizePolicy().hasHeightForWidth())
        self.menu_btn.setSizePolicy(sizePolicy)
        self.menu_btn.setObjectName("menu_btn")
        self.menu_layout.addWidget(self.menu_btn)
        self.reset_btn = QtWidgets.QPushButton(self.horizontalLayoutWidget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.reset_btn.sizePolicy().hasHeightForWidth())
        self.reset_btn.setSizePolicy(sizePolicy)
        self.reset_btn.setObjectName("reset_btn")
        self.menu_layout.addWidget(self.reset_btn)

        self.assign_to_camera_btn = QtWidgets.QPushButton(self.manualControlPage)
        self.assign_to_camera_btn.setGeometry(QtCore.QRect(10, 380, 141, 32))
        self.assign_to_camera_btn.setObjectName("assign_to_camera_btn")
        self.assign_to_camera_btn.hide()

        self.unassign_to_camera_btn = QtWidgets.QPushButton(self.manualControlPage)
        self.unassign_to_camera_btn.setGeometry(QtCore.QRect(10, 380, 141, 32))
        self.unassign_to_camera_btn.setObjectName("unassign_to_camera_btn")
        self.unassign_to_camera_btn.hide()

        # Button Commands
        self.up_left_btn.clicked.connect(self.move_left_up)
        self.up_btn.clicked.connect(self.move_up)
        self.up_right_btn.clicked.connect(self.move_right_up)
        self.left_btn.clicked.connect(self.move_left)
        self.right_btn.clicked.connect(self.move_right)
        self.down_left_btn.clicked.connect(self.move_left_down)
        self.down_btn.clicked.connect(self.move_down)
        self.down_right_btn.clicked.connect(self.move_right_down)
        self.home_btn.clicked.connect(self.move_home)
        self.zoom_in_btn.clicked.connect(self.zoom_in)
        self.zoom_out_btn.clicked.connect(self.zoom_out)
        self.menu_btn.clicked.connect(self.menu)
        self.reset_btn.clicked.connect(self.reset)
        self.assign_to_camera_btn.clicked.connect(self.assign_ptz_dialog)
        self.unassign_to_camera_btn.clicked.connect(self.unassign_ptz)

        self.formTabWidget.addTab(self.manualControlPage, "")
        self.gridLayout.addWidget(self.formTabWidget, 0, 0, 3, 1)

        self.shown_cameras = QtWidgets.QWidget()
        self.flowLayout = FlowLayout()
        self.flowLayout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        self.shown_cameras.setLayout(self.flowLayout)
        self.shown_cameras.setSizePolicy(sizePolicy)
        self.gridLayout.addWidget(self.shown_cameras, 0, 1, 1, 1)

        # handling camera window sizing
        self.screen_width = QtWidgets.QApplication.desktop().screenGeometry().width()
        self.screen_height = QtWidgets.QApplication.desktop().screenGeometry().height()

        # Menu
        AutoPTZ.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(AutoPTZ)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 705, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSource = QtWidgets.QMenu(self.menubar)
        self.menuSource.setObjectName("menuSource")
        self.menuAdd_Face = QtWidgets.QMenu(self.menubar)
        self.menuAdd_Face.setObjectName("menuAdd_Face")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        AutoPTZ.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(AutoPTZ)
        self.statusbar.setObjectName("statusbar")
        AutoPTZ.setStatusBar(self.statusbar)
        self.actionSave = QtWidgets.QAction(AutoPTZ)
        self.actionSave.setObjectName("actionSave")
        self.actionSave_as = QtWidgets.QAction(AutoPTZ)
        self.actionSave_as.setObjectName("actionSave_as")
        self.actionSave_as_2 = QtWidgets.QAction(AutoPTZ)
        self.actionSave_as_2.setObjectName("actionSave_as_2")
        self.actionClose = QtWidgets.QAction(AutoPTZ)
        self.actionClose.setObjectName("actionClose")
        self.actionAdd_IP = QtWidgets.QAction(AutoPTZ)

        self.actionAdd_IP.setObjectName("actionAdd_IP")
        self.menuAdd_NDI = QtWidgets.QMenu(AutoPTZ)
        self.menuAdd_NDI.setObjectName("menuAdd_NDI")
        self.menuAdd_Hardware = QtWidgets.QMenu(AutoPTZ)
        self.menuAdd_Hardware.setObjectName("menuAdd_Hardware")

        self.getNDISourceList()
        self.getPhysicalSourcesList()

        self.actionEdit = QtWidgets.QAction(AutoPTZ)
        self.actionEdit.setObjectName("actionEdit")
        self.actionContact = QtWidgets.QAction(AutoPTZ)
        self.actionContact.setObjectName("actionContact")
        self.actionAbout = QtWidgets.QAction(AutoPTZ)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout_2 = QtWidgets.QAction(AutoPTZ)
        self.actionAbout_2.setObjectName("actionAbout_2")
        self.actionContact_2 = QtWidgets.QAction(AutoPTZ)
        self.actionContact_2.setObjectName("actionContact_2")
        self.actionFacial_Recognition = QtWidgets.QAction(AutoPTZ)
        self.actionFacial_Recognition.setObjectName("actionFacial_Recognition")
        self.actionTrain_Model = QtWidgets.QAction(AutoPTZ)
        self.actionTrain_Model.setObjectName("actionTrain_Model")
        self.actionRemove_Face = QtWidgets.QAction(AutoPTZ)
        self.actionRemove_Face.setObjectName("actionRemove_Face")
        self.actionReset_Database = QtWidgets.QAction(AutoPTZ)
        self.actionReset_Database.setObjectName("actionReset_Database")
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave_as)
        self.menuFile.addAction(self.actionSave_as_2)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionClose)
        self.menuSource.addAction(self.actionAdd_IP)
        self.menuSource.addMenu(self.menuAdd_NDI)
        self.menuSource.addMenu(self.menuAdd_Hardware)
        self.menuSource.addSeparator()
        self.menuSource.addAction(self.actionEdit)
        self.menuAdd_Face.addAction(self.actionFacial_Recognition)
        self.menuAdd_Face.addAction(self.actionTrain_Model)
        self.menuAdd_Face.addAction(self.actionRemove_Face)
        self.menuAdd_Face.addSeparator()
        self.menuAdd_Face.addAction(self.actionReset_Database)
        self.menuHelp.addAction(self.actionAbout_2)
        self.menuHelp.addSeparator()
        self.menuHelp.addAction(self.actionContact_2)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSource.menuAction())
        self.menubar.addAction(self.menuAdd_Face.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(AutoPTZ)
        QtCore.QMetaObject.connectSlotsByName(AutoPTZ)

    def assign_ptz_dialog(self):
        """Launch the employee dialog."""
        if not camera_widget_list or self.select_camera_dropdown.currentText() == "":
            print("Need to select or add a camera")
        else:
            dlg = AssignCamDlg(self, cameraList=camera_widget_list, assignedList=assigned_ptz_camera,ptz_id=self.select_camera_dropdown.currentText())
            dlg.closeEvent = self.refreshBtnOnClose
            dlg.exec()

    def unassign_ptz(self):
        index = self.assigned_ptz_camera.index(self.select_camera_dropdown.currentText())

        camera = self.assigned_ptz_camera[index+1]
        camera.set_tracker(None)
        self.assigned_ptz_camera.remove(camera)
        self.assigned_ptz_camera.remove(self.select_camera_dropdown.currentText())

        self.unassign_to_camera_btn.hide()
        self.assign_to_camera_btn.show()

    def refreshBtnOnClose(self, event):
        if self.select_camera_dropdown.currentText() in self.assigned_ptz_camera:
            self.unassign_to_camera_btn.show()
        else:
            self.assign_to_camera_btn.show()

    def getPhysicalSourcesList(self):
        """
        Test the ports and returns a tuple with the available ports and the ones that are working.
        """
        non_working_ports = []
        dev_port = 0
        working_ports = []
        available_ports = []
        while len(non_working_ports) < 6:  # if there are more than 5 non working ports stop the testing.
            camera = cv2.VideoCapture(dev_port)
            if not camera.isOpened():
                non_working_ports.append(dev_port)
                # print("Port %s is not working." % dev_port)
            else:
                is_reading, img = camera.read()
                w = camera.get(3)
                h = camera.get(4)
                if is_reading:
                    working_ports.append(dev_port)
                    self.addPhysicalSource("Camera %s (%s x %s)" % (dev_port + 1, w, h), dev_port)
                    # working_ports["Camera %s (%s x %s)" % (dev_port, w, h)] = str(dev_port)
                else:
                    # print("Port %s for camera ( %s x %s) is present but does not reads." % (dev_port, w, h))
                    available_ports.append(dev_port)
            dev_port += 1
        return

    def init_manual_control(self, device):
        global current_manual_device
        try:
            current_manual_device = camera.D100(device)
            current_manual_device.init()
            print("Camera Initialized")

            if device in self.assigned_ptz_camera:
                self.unassign_to_camera_btn.show()
            else:
                self.assign_to_camera_btn.show()
        except:
            self.assign_to_camera_btn.hide()
            self.unassign_to_camera_btn.hide()
            print("Please initialize another camera")

    def move_up(self):
        global current_manual_device
        try:
            current_manual_device.up(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_left(self):
        global current_manual_device
        try:
            current_manual_device.left(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_right(self):
        global current_manual_device
        try:
            current_manual_device.right(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_down(self):
        global current_manual_device
        try:
            current_manual_device.down(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_left_up(self):
        global current_manual_device
        try:
            current_manual_device.left_up(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_right_up(self):
        self.window_width = self.shown_cameras.window().width()
        self.window_height = self.shown_cameras.window().height()
        self.sizeFind()
        global current_manual_device
        try:
            current_manual_device.right_up(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_left_down(self):
        global current_manual_device
        try:
            current_manual_device.left_down(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_right_down(self):
        global current_manual_device
        try:
            current_manual_device.right_down(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_home(self):
        global current_manual_device
        try:
            current_manual_device.home()
            S = threading.Timer(3, self.move_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def move_stop(self):
        global current_manual_device
        try:
            current_manual_device.stop()
        except:
            print("Please initialize a camera")

    def menu(self):
        global current_manual_device
        try:
            current_manual_device.menu()
        except:
            print("Please initialize a camera")

    def zoom_in(self):
        global current_manual_device
        try:
            current_manual_device.zoom_in()
            S = threading.Timer(0.5, self.zoom_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def zoom_out(self):
        global current_manual_device
        try:
            current_manual_device.zoom_out()
            S = threading.Timer(0.5, self.zoom_stop)
            S.start()
        except:
            print("Please initialize a camera")

    def zoom_stop(self):
        global current_manual_device
        try:
            current_manual_device.zoom_stop()
        except:
            print("Please initialize a camera")

    def reset(self):
        global current_manual_device
        try:
            current_manual_device.reset()
        except:
            print("Please initialize a camera")

    def sizeFind(self):
        print(self.window_width, self.window_height)

    def getNDISourceList(self):
        self.sourceList = get_ndi_sources()

        for i, s in enumerate(self.sourceList):
            self.addNDISource(s)
            # self.sourceWidgets.addItem(QtWidgets.QListWidgetItem('%s. %s' % (i + 1, s.ndi_name)))

    def addNDISource(self, ndi_source_id):
        ndiSource = QtWidgets.QAction(ndi_source_id.ndi_name, self)
        ndiSource.setCheckable(True)
        ndiSource.triggered.connect(lambda: self.addCamera(-1, ndi_source_id, ndiSource))
        self.menuAdd_NDI.addAction(ndiSource)

    def addPhysicalSource(self, sourceName, sourceNumber):
        cameraSource = QtWidgets.QAction(sourceName, self)
        cameraSource.setCheckable(True)
        cameraSource.triggered.connect(lambda: self.addCamera(sourceNumber, menuItem=cameraSource, ndi_source=None))
        self.menuAdd_Hardware.addAction(cameraSource)

    def addCamera(self, source, ndi_source, menuItem):
        cameraWidget = QtWidgets.QWidget()

        if source == -1:
            camera = NDICameraWidget(self.screen_width // 3, self.screen_height // 3, ndi_source=ndi_source,
                                     aspect_ratio=True)
            camera.setObjectName('NDI Camera: ' + ndi_source.ndi_name)
            cameraWidget.setObjectName('NDI Camera: ' + ndi_source.ndi_name)
            menuItem.disconnect()
            menuItem.triggered.connect(
                lambda: self.deleteCameraSource(source=-1, ndi_source=ndi_source, menuItem=menuItem, camera=camera,
                                                cameraWidget=cameraWidget))
        else:
            camera = CameraWidget(self.screen_width // 3, self.screen_height // 3, source, aspect_ratio=True)
            camera.setObjectName('Camera: ' + str(source + 1))
            cameraWidget.setObjectName('Camera ' + str(source + 1))
            menuItem.disconnect()
            menuItem.triggered.connect(
                lambda: self.deleteCameraSource(source=source, menuItem=menuItem,
                                                ndi_source=None, camera=camera, cameraWidget=cameraWidget))
            camera_widget_list.append(camera)
        # create internal grid layout for camera

        cameragridLayout = QtWidgets.QGridLayout()
        cameragridLayout.setObjectName('Camera Grid: ' + str(camera))

        cameragridLayout.addWidget(camera.get_video_frame(), 0, 0, 1, 1)
        cameraWidget.setLayout(cameragridLayout)
        select_cam_btn = QtWidgets.QPushButton("Select Camera")
        select_cam_btn.clicked.connect(lambda: camera.get_tracker())
        cameragridLayout.addWidget(select_cam_btn, 1, 0, 1, 1)

        self.flowLayout.addWidget(cameraWidget)

    def deleteCameraSource(self, source, ndi_source, menuItem, camera, cameraWidget):

        self.flowLayout.removeWidget(cameraWidget)
        camera.deleteLater()

        menuItem.disconnect()
        if source == -1:
            menuItem.triggered.connect(lambda: self.addCamera(source=-1, ndi_source=ndi_source, menuItem=menuItem))
            camera_widget_list.remove(cameraWidget)
        else:
            menuItem.triggered.connect(lambda: self.addCamera(source=source, ndi_source=None, menuItem=menuItem))

    def retranslateUi(self, AutoPTZ):
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
        self.assign_to_camera_btn.setText(_translate("AutoPTZ", "Assign PTZ"))
        self.unassign_to_camera_btn.setText(_translate("AutoPTZ", "Unassign PTZ"))
        self.formTabWidget.setTabText(self.formTabWidget.indexOf(self.manualControlPage),
                                      _translate("AutoPTZ", "Manual"))
        self.menuFile.setTitle(_translate("AutoPTZ", "File"))
        self.menuSource.setTitle(_translate("AutoPTZ", "Sources"))
        self.menuAdd_Face.setTitle(_translate("AutoPTZ", "Facial Recognition"))
        self.menuHelp.setTitle(_translate("AutoPTZ", "Help"))
        self.actionSave.setText(_translate("AutoPTZ", "Open"))
        self.actionSave_as.setText(_translate("AutoPTZ", "Save"))
        self.actionSave_as_2.setText(_translate("AutoPTZ", "Save as"))
        self.actionClose.setText(_translate("AutoPTZ", "Close"))
        self.actionAdd_IP.setText(_translate("AutoPTZ", "Add IP"))
        self.menuAdd_NDI.setTitle(_translate("AutoPTZ", "Add NDI"))
        self.menuAdd_Hardware.setTitle(_translate("AutoPTZ", "Add Hardware"))
        self.actionEdit.setText(_translate("AutoPTZ", "Edit Setup"))
        self.actionContact.setText(_translate("AutoPTZ", "Contact"))
        self.actionAbout.setText(_translate("AutoPTZ", "About"))
        self.actionAbout_2.setText(_translate("AutoPTZ", "About"))
        self.actionContact_2.setText(_translate("AutoPTZ", "Contact"))
        self.actionFacial_Recognition.setText(_translate("AutoPTZ", "Add Face"))
        self.actionTrain_Model.setText(_translate("AutoPTZ", "Train Model"))
        self.actionRemove_Face.setText(_translate("AutoPTZ", "Remove Face"))
        self.actionReset_Database.setText(_translate("AutoPTZ", "Reset Database"))
