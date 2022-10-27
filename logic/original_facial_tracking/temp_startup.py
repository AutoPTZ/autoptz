import sys

from PyQt5 import QtCore, QtWidgets as qtw
from PyQt5.QtCore import QTimer

from logic.old_facial_tracking.face_recognition_tracker_dlib import face_object_track_dlib
# from logic.original_facial_tracking.old.org_face_recognition_tracker import face_object_track
from logic.old_facial_tracking.facial_recognition import register_person, recognize_face


# def continue_click(self):
#     print('Selected: ' + self.sourceWidgets.currentItem().text())
#     drawFrame(self.sourceList[self.sourceWidgets.currentRow()])

def new_face_click():
    register_person()


def run_face_recognition_click():
    recognize_face()


def run_adv_face_track_click():
    face_object_track_dlib()


# UI Creation
class startAutoNDI(qtw.QWidget):
    def __init__(self):
        super(startAutoNDI, self).__init__()
        self.continue_btn = None
        self.new_face_btn = None
        self.face_recognition_btn = None
        self.adv_face_tracker_btn = None
        self.sourceWidgets = qtw.QListWidget()
        self.sourceList = []
        self.timer = None
        self.middleBodyLayout = None
        self.sizePolicy = None
        self.refresh_btn = None
        self.title = None
        self.topBarLayout = None
        self.search_ndi()

    def search_ndi(self):
        self.setWindowTitle("AutoPTZ")
        self.setLayout(qtw.QVBoxLayout())

        # Add widgets to layout

        # Top Bar
        self.topBarLayout = qtw.QHBoxLayout()
        self.title = qtw.QLabel("Found NDI Sources")
        self.refresh_btn = qtw.QPushButton()
        self.refresh_btn.setIcon(self.style().standardIcon(qtw.QStyle.StandardPixmap.SP_BrowserReload))
        self.sizePolicy = qtw.QSizePolicy(qtw.QSizePolicy.Fixed, qtw.QSizePolicy.Fixed)
        self.sizePolicy.setHorizontalStretch(0)
        self.sizePolicy.setVerticalStretch(0)
        self.sizePolicy.setHeightForWidth(self.refresh_btn.sizePolicy().hasHeightForWidth())
        self.refresh_btn.setSizePolicy(self.sizePolicy)
        self.refresh_btn.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.refresh_btn.setIconSize(QtCore.QSize(16, 16))
        self.refresh_btn.clicked.connect(qtw.QWidget.update)
        self.topBarLayout.addWidget(self.title)
        self.topBarLayout.addWidget(self.refresh_btn)

        # Middle NDI List
        self.middleBodyLayout = qtw.QVBoxLayout()
        self.timer = QTimer()
        self.timer.start(2000)
        # self.sourceList = get_ndi_sources()

        for i, s in enumerate(self.sourceList):
            self.sourceWidgets.addItem(qtw.QListWidgetItem('%s. %s' % (i + 1, s.ndi_name)))

        self.continue_btn = qtw.QPushButton("Old Method (Doesn't Do Anything)", self)
        # self.continue_btn.clicked.connect(self.continue_click)

        self.new_face_btn = qtw.QPushButton("Add New Face", self)
        self.new_face_btn.clicked.connect(new_face_click)

        self.face_recognition_btn = qtw.QPushButton("Run Simple Recognition", self)
        self.face_recognition_btn.clicked.connect(run_face_recognition_click)

        self.adv_face_tracker_btn = qtw.QPushButton("Run Advanced Tracker", self)
        self.adv_face_tracker_btn.clicked.connect(face_object_track_dlib)

        self.middleBodyLayout.addWidget(self.sourceWidgets)
        self.middleBodyLayout.addWidget(self.continue_btn)
        self.middleBodyLayout.addWidget(self.new_face_btn)
        self.middleBodyLayout.addWidget(self.face_recognition_btn)
        self.middleBodyLayout.addWidget(self.adv_face_tracker_btn)

        # Add Layouts Together
        self.layout().addLayout(self.topBarLayout)
        self.layout().addLayout(self.middleBodyLayout)

        self.show()


def main():
    app = qtw.QApplication(sys.argv)
    ex = startAutoNDI()
    # Runner
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
