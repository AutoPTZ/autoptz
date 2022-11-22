from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
import cv2
import time
import imutils
import shared.constants as constants
from logic.facial_tracking.testing_image_processor import ImageProcessor
from views.widgets.video_thread import VideoThread


# For now this only creates USB Camera Widget
class CameraWidget(QLabel):
    change_selection_signal = Signal()
    # FPS for Performance
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0

    def __init__(self, source, width, height, isNDI=False):
        super().__init__()
        self.width = width
        self.height = height
        self.setProperty('active', False)
        # self.resize(width, height)
        self.setObjectName(f"Camera Source: {source}")
        self.setStyleSheet(constants.CAMERA_STYLESHEET)
        self.setText(f"Camera Source: {source}")
        self.mouseReleaseEvent = lambda event, widget=self: self.clicked_widget(event, widget)

        # Create Video Capture Thread
        self.stream_thread = VideoThread(src=source, width=width, isNDI=isNDI)
        # Connect it's Signal to the update_image Slot Method
        self.stream_thread.change_pixmap_signal.connect(self.update_image)
        # Start the Thread
        self.stream_thread.start()

        # Create and Run Image Processor Thread
        self.processor_thread = ImageProcessor(stream_thread=self.stream_thread)
        self.processor_thread.start()

    def stop(self):
        self.processor_thread.stop()
        self.stream_thread.stop()
        self.deleteLater()

    def set_add_name(self, name):
        print(self.processor_thread.is_alive())
        if self.processor_thread.is_alive():
            self.processor_thread.add_name = name
        else:
            print(f"starting ImageProcessor Thread for {self.objectName()}")
            # Create and Run Image Processor Thread
            self.processor_thread = ImageProcessor(stream_thread=self.stream_thread)
            self.processor_thread.add_name = name
            self.processor_thread.start()

    def check_encodings(self):
        if self.processor_thread.is_alive():
            self.processor_thread.check_encodings()
        else:
            print(f"starting ImageProcessor Thread for {self.objectName()}")
            self.processor_thread = ImageProcessor(stream_thread=self.stream_thread)
            self.processor_thread.start()

    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image and draws on latest frame if processing is completed"""
        cv_img = self.draw_on_face(frame=cv_img, face_locations=self.processor_thread.face_locations,
                                   face_names=self.processor_thread.face_names,
                                   confidence_list=self.processor_thread.confidence_list)
        qt_img = self.convert_cv_qt(cv_img)
        self.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_qt_format.scaled(self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def clicked_widget(self, event, widget):
        if constants.CURRENT_ACTIVE_CAM_WIDGET is not None:
            constants.CURRENT_ACTIVE_CAM_WIDGET.setProperty(
                'active', not constants.CURRENT_ACTIVE_CAM_WIDGET.property('active'))
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()

        if constants.CURRENT_ACTIVE_CAM_WIDGET == widget:
            constants.CURRENT_ACTIVE_CAM_WIDGET = None
        else:
            constants.CURRENT_ACTIVE_CAM_WIDGET = widget
            constants.CURRENT_ACTIVE_CAM_WIDGET.setProperty(
                'active', not constants.CURRENT_ACTIVE_CAM_WIDGET.property('active'))
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()
        self.change_selection_signal.emit()

    def draw_on_face(self, frame, face_locations, face_names, confidence_list):
        if face_locations is not None and face_names is not None and confidence_list is not None:
            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidence_list):
                # Scale back up face locations since the frame we detected in was scaled to 1/2 size
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                # Draw a label with name and confidence for the face
                cv2.putText(frame, name, (left + 5, top - 5), constants.FONT, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, confidence, (right - 52, bottom - 5), constants.FONT, 0.45, (255, 255, 0), 1)
        # FPS Counter
        self.fc += 1
        time_set = time.time() - self.start_time
        if time_set >= self.display_time:
            self.FPS = self.fc / time_set
            self.fc = 0
            self.start_time = time.time()
        fps = "FPS: " + str(self.FPS)[:5]

        cv2.putText(frame, fps, (50, 50), constants.FONT, 1, (0, 0, 255), 2)
        return frame

    def closeEvent(self, event):
        self.processor_thread.stop()
        self.stream_thread.stop()
        event.accept()

    """
             OpenCV VideoThread -> sent for processing (which easily uses OpenCV frame)

             QT uses QPixmap, so we need to convert OpenCV frame to QPixmap

             FrameProcess == Facial Recognition, Tracking, etc
             VideoThread == Turns on Camera and constantly gets frames
             CameraWidget == Shows the frames on QT to the user

             VideoThread -> FrameProcess (sends back, boxes + names)

             VideoThread -> Pixmap -> CameraWidget
             VideoThread -> CameraWidget -> Pixmap


             VideoThread ->  CameraWidget  <- FrameProcess (boxes)
             CameraWidget -> Pixmap

            """
