from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtCore import Signal
import cv2
import time
import dlib
import NDIlib as ndi
from logic.facial_recognition.facial_recognition import FacialRecognition
import shared.constants as constants
from views.widgets.video_thread import VideoThread
from multiprocessing import Process, Manager, Queue, Value


def run_facial_recognition(frame_queue, shared_data, facial_recognition, stop_signal):

    while not stop_signal.value:
        # Get the latest frame from the queue
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Run the facial recognition on the frame
            face_locations, face_names, confidence_list = facial_recognition.recognize(
                frame)

            # Update the facial recognition results for this camera
            shared_data['facial_recognition_results'] = (
                face_locations, face_names, confidence_list)


class CameraWidget(QLabel):
    """
    Create and handle all Cameras that are added to the UI.
    It creates a QLabel as OpenCV and NDI video can be converted to QPixmap for display.
    Combines both VideoThread and Facial Recognition Processes for asynchronous computation for smoother looking video.
    With the latest frame, Dlib Object Tracking is in use and works alongside with Face Recognition to fix any inconsistencies when tracking.
    """
    change_selection_signal = Signal()
    start_time = time.time()
    display_time = 2
    fc = 0
    FPS = 0
    stream_thread = None
    track_started = None
    temp_tracked_name = None
    track_x = None
    track_y = None
    track_w = None
    track_h = None
    is_tracking = None
    tracked_name = None
    is_moving = False

    def __init__(self, source, width, height, lock, isNDI=False):
        super().__init__()
        self.width = width
        self.height = height
        self.lock = lock
        self.isNDI = isNDI
        self._is_stopped = True
        self.setProperty('active', False)
        if self.isNDI:
            self.setObjectName(f"Camera Source: {source.ndi_name}")
        else:
            self.setObjectName(f"Camera Source: {source}")
        self.setStyleSheet(constants.CAMERA_STYLESHEET)
        self.setText(f"Camera Source: {source}")
        self.mouseReleaseEvent = lambda event, widget=self: self.clicked_widget(
            event, widget)

        # Create a Queue to hold the latest frame
        self.frame_queue = Queue(maxsize=1)
        self.stop_signal = Value('b', False)

        # Create Video Capture Thread
        self.stream_thread = VideoThread(src=source, width=width, isNDI=isNDI)
        # Connect it's Signal to the update_image Slot Method
        self.stream_thread.change_pixmap_signal.connect(
            self.update_image_and_queue)

        # Start the Thread
        self.stream_thread.start()

        self.manager = Manager()
        self.shared_data = self.manager.dict()
        self.shared_data['facial_recognition_results'] = ([], [], [])
        self.shared_data['add_face_name'] = None

        self.facial_recognition = FacialRecognition(
            self.shared_data)

        # Create and start a facial recognition process for this camera
        self.facial_recognition_process = Process(
            target=run_facial_recognition,
            args=(self.frame_queue, self.shared_data, self.facial_recognition,
                  self.stop_signal)
        )
        self.facial_recognition_process.start()

        # PTZ Movement Thread
        self.last_request = None
        if isNDI and ndi.recv_ptz_is_supported(instance=self.stream_thread.ndi_recv):
            print(f"This NDI Source {source.ndi_name} Supports PTZ Movement")
            self.ptz_controller = self.stream_thread.ndi_recv
        else:
            if isNDI:
                print(
                    f"This NDI Source {source.ndi_name} Does NOT Supports PTZ Movement")
            self.ptz_controller = None
        self.ptz_is_usb = None

        self.track_started = False
        self.temp_tracked_name = None
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        self.is_tracking = False  # If Track Checkbox is checked
        self.tracked_name = None  # Face that needs to be tracked
        self.tracker = dlib.correlation_tracker()

    def stop(self):
        """
        When CameraWidget is being removed from the UI, we should stop associated PTZ cameras and all threads before deletion.
        """
        if self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                ndi.recv_ptz_pan_tilt_speed(
                    instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
        self.stop_signal.value = True
        self.facial_recognition_process.join()
        self.facial_recognition_process.kill()
        self.stream_thread.stop()
        self.deleteLater()
        self.destroy()

    def set_ptz(self, control, isUSB=False):
        """
        Sets PTZ controller from Homepage
        If control is None then first stop PTZ and disconnect from it
        :param control:
        :param isUSB:
        """
        if control is None and self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                self.ptz_controller.pantilt(pan_speed=0, tilt_speed=0)
                self.ptz_controller.close_connection()
        self.ptz_controller = control
        self.ptz_is_usb = isUSB

    def set_tracked_name(self, name):
        """
        Sets the name to track, used to start tracking based on face recognition data
        :param name:
        """
        self.track_started = False
        self.tracked_name = name

    def reset_tracking(self):
        """
        Resets tracking variables when user toggles the checkbox
        """
        self.track_x = None
        self.track_y = None
        self.track_w = None
        self.track_h = None
        self.is_tracking = not self.is_tracking
        if self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                ndi.recv_ptz_pan_tilt_speed(
                    instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
            self.last_request = None

    def update_image_and_queue(self, cv_img):
        """Updates the QLabel with the latest OpenCV/NDI frame and draws it"""
        cv_img = self.draw_on_frame(frame=cv_img)
        qt_img = self.convert_cv_qt(cv_img)
        self.setPixmap(qt_img)
        """Add latest frame to queue when possible"""
        if not self.frame_queue.full():
            self.frame_queue.put(cv_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_qt_format = QtGui.QImage(
            rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        p = convert_to_qt_format.scaled(
            self.width, self.height, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def clicked_widget(self, event, widget):
        """
        First checks if there is another CameraWidget currently active. If it is then deactivate it and update their stylesheet.
        Then if that deactivated CameraWidget is the same CameraWidget currently clicked on, then remove it from constants. So nothing is active.
        If it is not the same CameraWidget, then update this clicked on CameraWidget to be active and update its stylesheet.
        :param event:
        :param widget:
        """
        if constants.CURRENT_ACTIVE_CAM_WIDGET is not None:
            constants.CURRENT_ACTIVE_CAM_WIDGET.setProperty(
                'active', not constants.CURRENT_ACTIVE_CAM_WIDGET.property('active'))
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()

        if constants.CURRENT_ACTIVE_CAM_WIDGET == widget:
            constants.CURRENT_ACTIVE_CAM_WIDGET = None
        else:
            constants.CURRENT_ACTIVE_CAM_WIDGET = widget
            constants.CURRENT_ACTIVE_CAM_WIDGET.setProperty(
                'active', not constants.CURRENT_ACTIVE_CAM_WIDGET.property('active'))
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().unpolish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.style().polish(
                constants.CURRENT_ACTIVE_CAM_WIDGET)
            constants.CURRENT_ACTIVE_CAM_WIDGET.update()
        self.change_selection_signal.emit()

    def draw_on_frame(self, frame):
        """
        Is called by update_image and returns the latest frame with FPS + face box drawings if there are any.
        :param frame:
        :return:
        """
        print(
            self.shared_data['facial_recognition_results'])
        if self.shared_data['facial_recognition_results'] != ([], [], []):
            face_locations, face_names, confidence_list = self.shared_data[
                'facial_recognition_results']

            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidence_list):
                if name == self.tracked_name:
                    self.temp_tracked_name = name
                    self.track_x = left
                    self.track_y = top
                    self.track_w = right
                    self.track_h = bottom
                # Draw a box around the face
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 255, 0), 2)
                # Draw a label with name and confidence for the face
                cv2.putText(frame, name, (left + 5, top - 5),
                            constants.FONT, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, confidence, (right - 52, bottom - 5),
                            constants.FONT, 0.45, (255, 255, 0), 1)

        if self.is_tracking and self.track_x is not None and self.track_y is not None and self.track_w is not None and self.track_h is not None:
            frame = self.track_face(
                frame, self.track_x, self.track_y, self.track_w, self.track_h)
            # frame = self.track_face(frame, self.track_x - 2, self.track_y - 5, self.track_w + 8, self.track_h + +15)
        self.temp_tracked_name = None

        # FPS Counter
        self.fc += 1
        time_set = time.time() - self.start_time
        if time_set >= self.display_time:
            self.FPS = self.fc / time_set
            self.fc = 0
            self.start_time = time.time()
        fps = "FPS: " + str(self.FPS)[:5]

        cv2.putText(frame, fps, (20, 30), constants.FONT, 0.7, (0, 0, 255), 2)
        return frame

    def track_face(self, frame, x, y, w, h):
        """
        Uses Dlib Object Tracking to set and update the currently tracked person
        Then if a PTZ camera is associated, it should move the camera in any direction automatically
        :param frame:
        :param x:
        :param y:
        :param w:
        :param h:
        :return:
        """
        min_x = int(frame.shape[1] / 9.2)
        max_x = int(frame.shape[1] / 1.2)
        min_y = int(frame.shape[0] / 18)
        max_y = int(frame.shape[0] / 1.6)
        cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.putText(frame, f"TRACKING {self.tracked_name.upper()}",
                    (20, 52), constants.FONT, 0.7, (0, 0, 255), 2)

        if self.track_started is False:
            rect = dlib.rectangle(x, y, w, h)
            self.tracker.start_track(rgb_frame, rect)
            cv2.putText(frame, "tracking", (x, h + 15),
                        constants.FONT, 0.45, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 3, 1)
            self.track_started = True
        if self.temp_tracked_name == self.tracked_name:
            rect = dlib.rectangle(x, y, w, h)
            self.tracker.start_track(rgb_frame, rect)
            cv2.putText(frame, "tracking", (x, h + 15),
                        constants.FONT, 0.45, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 3, 1)
        else:
            self.tracker.update(rgb_frame)
            pos = self.tracker.get_position()
            # unpack the position object
            x = int(pos.left())
            y = int(pos.top())
            w = int(pos.right())
            h = int(pos.bottom())
            cv2.putText(frame, "tracking", (x, h + 15),
                        constants.FONT, 0.45, (0, 255, 0), 1)
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 3, 1)

        if self.ptz_controller is not None:
            if x > min_x and w < max_x and y > min_y and h < max_y and self.last_request != "stop":
                if self.ptz_is_usb:
                    self.ptz_controller.move_stop()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
                    # self.ptz_controller.pantilt(pan_speed=0, tilt_speed=0)
                self.last_request = "stop"

            if y < min_y and x < min_x and self.last_request != "up_left":
                if self.ptz_is_usb:
                    self.ptz_controller.move_left_up_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=0.1, tilt_speed=0.1)
                    # self.ptz_controller.pantilt(pan_speed=2, tilt_speed=1)
                self.last_request = "up_left"

            elif y < min_y and w > max_x and self.last_request != "up_right":
                if self.ptz_is_usb:
                    self.ptz_controller.move_right_up_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=-0.1, tilt_speed=0.1)
                    # self.ptz_controller.pantilt(pan_speed=-2, tilt_speed=1)
                self.last_request = "up_right"

            elif h > max_y and x < min_x and self.last_request != "down_left":
                if self.ptz_is_usb:
                    self.ptz_controller.move_left_down_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=0.1, tilt_speed=-0.1)
                    # self.ptz_controller.pantilt(pan_speed=2, tilt_speed=-1)
                self.last_request = "down_left"

            elif h > max_y and w > max_x and self.last_request != "down_right":
                if self.ptz_is_usb:
                    self.ptz_controller.move_right_down_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=-0.1, tilt_speed=-0.1)
                    # self.ptz_controller.pantilt(pan_speed=-2, tilt_speed=-1)
                self.last_request = "down_right"

            elif y < min_y and self.last_request != "up":
                if self.ptz_is_usb:
                    self.ptz_controller.move_up_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=0, tilt_speed=0.1)
                    # self.ptz_controller.pantilt(pan_speed=0, tilt_speed=1)
                self.last_request = "up"

            elif h > max_y and self.last_request != "down":
                if self.ptz_is_usb:
                    self.ptz_controller.move_down_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=0, tilt_speed=-0.1)
                    # self.ptz_controller.pantilt(pan_speed=0, tilt_speed=-1)
                self.last_request = "down"

            elif x < min_x and self.last_request != "left":
                if self.ptz_is_usb:
                    self.ptz_controller.move_left_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=0.1, tilt_speed=0)
                    # self.ptz_controller.pantilt(pan_speed=2, tilt_speed=0)
                self.last_request = "left"

            elif w > max_x and self.last_request != "right":
                if self.ptz_is_usb:
                    self.ptz_controller.move_right_track()
                else:
                    ndi.recv_ptz_pan_tilt_speed(
                        instance=self.ptz_controller, pan_speed=-0.1, tilt_speed=0)
                    # self.ptz_controller.pantilt(pan_speed=-2, tilt_speed=0)
                self.last_request = "right"

        return frame
