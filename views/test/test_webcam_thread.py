import time
from PyQt6 import QtGui
from PyQt6.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout
from PyQt6.QtGui import QPixmap
import sys
import cv2
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            ret, cv_img = cap.read()
            self.fc += 1
            TIME = time.time() - self.start_time

            if (TIME) >= self.display_time:
                self.FPS = self.fc / (TIME)
                self.fc = 0
                self.start_time = time.time()

            fps = "FPS: " + str(self.FPS)[:5]
            if ret:
                cv2.putText(cv_img, fps, (50, 50), self.font, 1, (0, 0, 255), 2)
                self.change_pixmap_signal.emit(cv_img)
        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


class App(QWidget):
    defaultStyle = """
            QLabel[active="false"]{
                border: 2.5px solid slategray;
                border-radius: 3px;}

            QLabel::hover {
                border: 2.5px solid crimson;
                border-radius: 3px;}
                
            QLabel[active="true"]{
                border: 2.5px solid dodgerblue;
                border-radius: 3px;}
            """

    def __init__(self):
        super().__init__()
        self.current_select = None
        self.setWindowTitle("Qt live label demo")
        self.disply_width = 1280
        self.display_height = 720
        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setProperty('active', False)
        self.image_label.setStyleSheet(self.defaultStyle)
        self.image_label.resize(self.disply_width, self.display_height)
        self.image_label.mouseReleaseEvent = lambda event, label=self.image_label: self.clicked_widget(event, label)
        self.image_label.setObjectName("webcam image")
        # create a text label
        self.textLabel = QLabel('Webcam')
        self.textLabel.setProperty('active', False)
        self.textLabel.setStyleSheet(self.defaultStyle)
        self.textLabel.mouseReleaseEvent = lambda event, label=self.textLabel: self.clicked_widget(event, label)
        self.textLabel.setObjectName("Webcam bottom text")
        # create a vertical box layout and add the two labels
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)
        vbox.addWidget(self.textLabel)
        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.thread.start()
    
    def clicked_widget(self, event, label):
        if self.current_select is not None:
            self.current_select.setProperty(
                'active', not self.current_select.property('active'))
            self.current_select.style().unpolish(self.current_select)
            self.current_select.style().polish(self.current_select)
            self.current_select.update()

        if self.current_select == label:
            self.current_select = None
            return
        else:
            self.current_select = label
            self.current_select.setProperty(
                'active', not self.current_select.property('active'))
            self.current_select.style().unpolish(self.current_select)
            self.current_select.style().polish(self.current_select)
            self.current_select.update()
            print(self.current_select.objectName())

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)


if __name__=="__main__":
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    a = App()
    a.show()
    sys.exit(app.exec())
