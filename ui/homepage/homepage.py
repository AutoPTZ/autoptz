import cv2
from PyQt5 import QtCore, QtWidgets

from logic.camera_search.search_ndi import get_ndi_sources
from ui.widgets.camera_widget import CameraWidget
from ui.widgets.ndi_cam_widget import NDICameraWidget

countRow = 0
countCol = 1


class Ui_AutoPTZ(object):
    def setupUi(self, AutoPTZ):
        AutoPTZ.setObjectName("AutoPTZ")
        AutoPTZ.resize(705, 520)
        AutoPTZ.setMinimumSize(QtCore.QSize(705, 520))
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

        # Dynamically determine screen width/height
        self.screen_width = QtWidgets.QApplication.desktop().screenGeometry().width()
        self.screen_height = QtWidgets.QApplication.desktop().screenGeometry().height()
        # self.screen_width = self.gridLayout.geometry().width()
        # self.screen_height = self.gridLayout.geometry().height()

        self.formToolBox = QtWidgets.QToolBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.formToolBox.sizePolicy().hasHeightForWidth())
        self.formToolBox.setSizePolicy(sizePolicy)
        self.formToolBox.setObjectName("formToolBox")
        self.formToolBoxPage1 = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.formToolBoxPage1.sizePolicy().hasHeightForWidth())
        self.formToolBoxPage1.setSizePolicy(sizePolicy)
        self.formToolBoxPage1.setObjectName("formToolBoxPage1")
        self.formLayout = QtWidgets.QFormLayout(self.formToolBoxPage1)
        self.formLayout.setLabelAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setFormAlignment(QtCore.Qt.AlignLeading | QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.formLayout.setObjectName("formLayout")
        self.checkBox = QtWidgets.QCheckBox(self.formToolBoxPage1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setChecked(False)
        self.checkBox.setAutoRepeat(False)
        self.checkBox.setAutoExclusive(False)
        self.checkBox.setObjectName("checkBox")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.checkBox)
        self.label = QtWidgets.QLabel(self.formToolBoxPage1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.comboBox = QtWidgets.QComboBox(self.formToolBoxPage1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setObjectName("comboBox")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.comboBox)
        self.formToolBox.addItem(self.formToolBoxPage1, "")
        self.gridLayout.addWidget(self.formToolBox, 0, 0, 3, 1)
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

        # self.actionAdd_Hardware.triggered.connect(lambda: self.addCameraSource())
        # self.actionAdd_Hardware.triggered.connect(lambda: self.list_ports())

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

    def getNDISourceList(self):
        self.sourceList = get_ndi_sources()

        for i, s in enumerate(self.sourceList):
            self.addNDISource(s)
            #self.sourceWidgets.addItem(QtWidgets.QListWidgetItem('%s. %s' % (i + 1, s.ndi_name)))

    def addNDISource(self, ndi_source_id):
        ndiSource = QtWidgets.QAction(ndi_source_id.ndi_name, self)
        ndiSource.setCheckable(True)
        ndiSource.triggered.connect(lambda: self.addCamera(-1, ndi_source_id, ndiSource))
        self.menuAdd_NDI.addAction(ndiSource)

    def addPhysicalSource(self, sourceName, sourceNumber):
        cameraSource = QtWidgets.QAction(sourceName, self)
        cameraSource.setCheckable(True)
        cameraSource.triggered.connect(lambda: self.addCamera(sourceNumber, menuItem= cameraSource, ndi_source=None))
        self.menuAdd_Hardware.addAction(cameraSource)

    def addCamera(self, source, ndi_source, menuItem):
        global countRow
        global countCol

        if countRow == 3:
            countRow = 0
            countCol = countCol + 1

        if source == -1:
            camera = NDICameraWidget(self.screen_width // 3, self.screen_height // 3, ndi_source=ndi_source, aspect_ratio=True)
            camera.setObjectName('NDI ' + str(countRow) + ' ' + str(countCol))
            self.gridLayout.addWidget(camera.get_video_frame(), countRow, countCol, 1, 1)
            tempR = countRow
            tempC = countCol
            menuItem.disconnect()
            menuItem.triggered.connect(lambda: self.deleteCameraSource(source = -1, ndi_source = ndi_source, menuItem = menuItem, sourceRow= tempR, sourceCol = tempC))
        else:
            camera = CameraWidget(self.screen_width // 3, self.screen_height // 3, source, aspect_ratio=True)
            camera.setObjectName('Camera ' + str(countRow) + ' ' + str(countCol))
            self.gridLayout.addWidget(camera.get_video_frame(), countRow, countCol, 1, 1)
            tempR = countRow
            tempC = countCol
            menuItem.disconnect()
            menuItem.triggered.connect(lambda: self.deleteCameraSource(source, menuItem, tempR, tempC))

        countRow = countRow + 1

    def deleteCameraSource(self, source, ndi_source, menuItem, sourceRow, sourceCol):
        cameraWidget = self.gridLayout.itemAtPosition(sourceRow, sourceCol).widget()
        cameraWidget.deleteLater()
        menuItem.disconnect()
        if source == -1:
            menuItem.triggered.connect(lambda: self.addCamera(source=-1, ndi_source=ndi_source, menuItem=menuItem))
        else:
            menuItem.triggered.connect(lambda: self.addCamera(source=source, ndi_source=None, menuItem=menuItem))

    def retranslateUi(self, AutoPTZ):
        _translate = QtCore.QCoreApplication.translate
        AutoPTZ.setWindowTitle(_translate("AutoPTZ", "AutoPTZ"))
        self.checkBox.setText(_translate("AutoPTZ", "Enable Tracking"))
        self.label.setText(_translate("AutoPTZ", "Profile Select"))
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
