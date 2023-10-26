import math
import os
import dlib
from PySide6 import QtGui
from PySide6.QtWidgets import QLabel
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt, Signal, QTimer
import cv2
import time
import NDIlib as ndi
import numpy as np
import mediapipe as mp
from logic.camera_search.search_ndi import get_ndi_sources
from logic.image_processing.facial_recognition import FacialRecognition
import shared.constants as constants
from multiprocessing import Process, Value, Queue
import imutils


def run_body_pose_estimation(shared_frames, body_pose_queue, objectName, stop_signal):
    print(f"Body Pose Estimation service is starting for {objectName}")
    pose_estimator = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.75,
                                            smooth_landmarks=True, min_tracking_confidence=0.6)
    while not stop_signal.value:
        if shared_frames:
            frame = shared_frames[-1]
            # Estimate Pose
            if frame.shape[2] == 4:
                # Convert from BGR or BGRA to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
            results = pose_estimator.process(frame)
            if results.pose_landmarks:
                body_pose_queue.put(results.pose_landmarks)


def run_facial_recognition(shared_frames, facial_recognition, objectName, stop_signal):
    print(f"Facial Recognition service is starting for {objectName}")
    frame_count = 0
    while not stop_signal.value:
        frame_count += 1
        if frame_count % 240 == 0:
            if shared_frames:
                facial_recognition.recognize(shared_frames[-1])


def run_camera_stream(shared_frames, source, width, stop_signal, isNDI=False):
    if type(source) == int:
        print(f"Camera Stream service is starting for Camera Source: {source}")
        cap = cv2.VideoCapture(source)
        if os.name == 'nt':  # fixes Windows OpenCV resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 5000)
    else:
        sources = get_ndi_sources()
        source = next(
            (src for src in sources if src.ndi_name == source), None)
        # Create the NDIlib.NDIlib.Source object here using the shared information
        print(f"Camera Stream service is starting for {source.ndi_name}")
        ndi_recv_create = ndi.RecvCreateV3(source_to_connect_to=source)
        ndi_recv_create.color_format = ndi.RECV_COLOR_FORMAT_BGRX_BGRA
        ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_LOWEST
        ndi_recv = ndi.recv_create_v3(ndi_recv_create)
        ndi.recv_connect(ndi_recv, source)

    while not stop_signal.value:
        if type(source) == int:
            ret, img = cap.read()
            if ret:
                cv_img = imutils.resize(img, width)
                if len(shared_frames) >= 120:
                    shared_frames.pop(0)  # Remove the oldest frame
                shared_frames.append(cv_img)
        else:
            ret, v, _, _ = ndi.recv_capture_v2(ndi_recv, 1000)
            if ret == ndi.FRAME_TYPE_VIDEO:
                cv_img = np.copy(v.data)
                ndi.recv_free_video_v2(ndi_recv, v)
                cv_img = imutils.resize(cv_img, width)
                if len(shared_frames) >= 10:
                    shared_frames.pop(0)  # Remove the oldest frame
                shared_frames.append(cv_img)
    if isNDI:
        ndi.recv_destroy(ndi_recv)
        ndi.destroy()
    else:
        cap.release()


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
    temp_tracked_name = None
    track_x = None
    track_y = None
    track_w = None
    track_h = None
    is_tracking = None
    tracked_name = None
    is_moving = False
    face_center_x = None
    face_center_y = None
    model = None

    def __init__(self, source, width, height, manager, isNDI=False):
        super().__init__()
        self.width = width
        self.height = height
        self.isNDI = isNDI
        self.setProperty('active', False)
        self.setStyleSheet(constants.CAMERA_STYLESHEET)
        if self.isNDI:
            self.setText(f"Camera Source: {source.ndi_name}")
            self.setObjectName(f"Camera Source: {source.ndi_name}")
        else:
            self.setText(f"Camera Source: {source}")
            self.setObjectName(f"Camera Source: {source}")
        self.mouseReleaseEvent = lambda event, widget=self: self.clicked_widget(
            event, widget)

        # PTZ Movement
        self.last_request = None
        self.ptz_controller = None

        if isNDI:
            ndi_recv_create = ndi.RecvCreateV3(source_to_connect_to=source)
            ndi_recv_create.bandwidth = ndi.RECV_BANDWIDTH_METADATA_ONLY
            ndi_recv = ndi.recv_create_v3(ndi_recv_create)
            ndi.recv_connect(ndi_recv, source)
            for i in range(2):
                ret, v, _, _ = ndi.recv_capture_v2(ndi_recv, 5000)
                if ndi.recv_ptz_is_supported(instance=ndi_recv):
                    print(
                        f"This NDI Source {source.ndi_name} Supports PTZ Movement")
                    self.ptz_controller = ndi_recv
        else:
            if isNDI:
                print(
                    f"This NDI Source {source.ndi_name} Does NOT Supports PTZ Movement")
            self.ptz_controller = None
        self.ptz_is_usb = None

        self.shared_camera_frames = manager.list()

        # Create a Queue objects
        self.facial_recognition_queue = Queue()
        self.body_pose_queue = Queue()

        # Signal to stop camera stream, facial recognition, and body pose processes
        self.stop_signal = Value('b', False)

        # Create and start the process
        if isNDI:
            self.camera_stream_process = Process(target=run_camera_stream, args=(
                self.shared_camera_frames, source.ndi_name, width, self.stop_signal, isNDI))
        else:
            self.camera_stream_process = Process(target=run_camera_stream, args=(
                self.shared_camera_frames, source, width, self.stop_signal, isNDI))
        self.camera_stream_process.start()

        # Create and start a Facial Recognition process
        self.facial_recognition = FacialRecognition(
            self.facial_recognition_queue, self.objectName())
        self.facial_recognition_process = Process(
            target=run_facial_recognition,
            args=(self.shared_camera_frames,
                  self.facial_recognition, self.objectName(), self.stop_signal)
        )
        self.restart_facial_recogntion()

        # Create and start the Body Pose Estimation process
        self.body_pose_estimation_process = Process(target=run_body_pose_estimation, args=(
            self.shared_camera_frames, self.body_pose_queue, self.objectName(), self.stop_signal))
        self.body_pose_estimation_process.start()

        # Start the QTimer to update the QLabel
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_image_and_queue)
        self.timer.start(1000 / 30)  # up to 30 fps

        self.temp_tracked_name = None
        self.tracker = None
        self.is_tracking = False  # If Track Checkbox is checked
        self.tracked_name = None  # Face that needs to be tracked

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
                ndi.recv_ptz_pan_tilt_speed(instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
                self.ptz_controller.close_connection()
        self.ptz_controller = control
        self.ptz_is_usb = isUSB

    def set_tracked_name(self, name):
        """
        Sets the name to track, used to start tracking based on face recognition data
        :param name:
        """
        self.tracked_name = name

    def reset_tracking(self):
        """
        Resets tracking variables when user toggles the checkbox
        """
        self.is_tracking = not self.is_tracking
        self.tracker = None
        if self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                ndi.recv_ptz_pan_tilt_speed(
                    instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
            self.last_request = None

    def restart_facial_recogntion(self):
        print(
            f"Facial Recognition service is restarting for {self.objectName()}")
        if self.facial_recognition_process.is_alive():
            self.facial_recognition_process.terminate()
            self.facial_recognition.check_encodings()
            self.facial_recognition_process = Process(
                target=run_facial_recognition,
                args=(self.shared_camera_frames,
                      self.facial_recognition, self.objectName(), self.stop_signal)
            )
        self.facial_recognition_process.start()

    def update_image_and_queue(self):
        """Updates the QLabel with the latest OpenCV/NDI frame and draws it"""
        if self.shared_camera_frames:
            # Get the latest frame without removing it
            cv_img = self.shared_camera_frames[-1]
            cv_img = self.draw_on_frame(frame=cv_img)
            qt_img = self.convert_cv_qt(cv_img)
            self.setPixmap(qt_img)

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

    def get_body_rectangle(self, body_pose, frame_width, frame_height):
        """
        Returns the bounding box of the head, shoulders, and waist.

        :param body_pose: The body pose landmarks.
        :return: A tuple of 4 integers representing the bounding box of the head, shoulders, and waist.
        """
        # Extract the landmarks from the body_pose
        landmarks = body_pose.landmark

        # Get the min and max x and y values of the landmarks
        min_x = min(landmarks[i].x for i in list(range(11)) + [23, 24])
        max_x = max(landmarks[i].x for i in list(range(11)) + [23, 24])
        min_y = min(landmarks[i].y for i in list(range(11)) + [23, 24])
        max_y = max(landmarks[i].y for i in list(range(11)) + [23, 24])

        # Convert the min and max values to pixel coordinates
        left = min_x * frame_width
        right = max_x * frame_width
        top = min_y * frame_height
        bottom = max_y * frame_height

        return int(left), int(top), int(right), int(bottom)

    def rectangles_overlap(self, rect1, rect2):
        # Calculate the intersection rectangle
        x1 = max(rect1[0], rect2[0])
        y1 = max(rect1[1], rect2[1])
        x2 = min(rect1[2], rect2[2])
        y2 = min(rect1[3], rect2[3])

        # Check if there is an intersection
        if x1 < x2 and y1 < y2:
            intersection_area = (x2 - x1) * (y2 - y1)
            rect1_area = (rect1[2] - rect1[0]) * (rect1[3] - rect1[1])
            rect2_area = (rect2[2] - rect2[0]) * (rect2[3] - rect2[1])
            min_area = min(rect1_area, rect2_area)
            # Check if one rectangle is entirely inside the other or if the intersection area is above a certain threshold
            if intersection_area >= 0.5 * min_area:
                return True

        return False

    def draw_on_frame(self, frame):
        centroid_x = None
        centroid_y = None

        # Get the facial recognition results
        face_details = ([], [], [])
        if not self.facial_recognition_queue.empty():
            face_details = self.facial_recognition_queue.get_nowait()

        # Get body pose results
        body_pose = None
        face_rectangle = None
        if not self.body_pose_queue.empty():
            body_pose = self.body_pose_queue.get_nowait()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        # Update the tracker
        if self.tracker and self.is_tracking:
            self.tracker.update(rgb_frame)

        # Get face rectangle
        if face_details != ([], [], []):
            face_locations, face_names, confidences = face_details
            for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidences):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                # Draw face rectangle and labels
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 255, 0), 2)

                # If this is the tracked face
                if name == self.tracked_name:
                    self.temp_tracked_name = name
                    face_rectangle = (left, top, right, bottom)

                # Draw face rectangle and labels
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left + 5, top - 5),
                            constants.FONT, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, str(confidence), (right - 52, bottom - 5),
                            constants.FONT, 0.45, (255, 255, 0), 1)

        # Get body rectangle
        if body_pose:
            frame_height, frame_width, _ = frame.shape
            body_rectangle = self.get_body_rectangle(
                body_pose, frame_width, frame_height)

            # Draw body rectangle
            cv2.rectangle(frame, (body_rectangle[0], body_rectangle[1]),
                          (body_rectangle[2], body_rectangle[3]), (0, 255, 0), 2)

            # Check if the body pose intersects or is near the face
            if face_rectangle and self.rectangles_overlap(body_rectangle, face_rectangle) and self.is_tracking:
                # Reinitialize the dlib tracker with the new body pose data
                self.tracker = dlib.correlation_tracker()
                self.tracker.start_track(
                    rgb_frame, dlib.rectangle(*body_rectangle))
            elif self.tracker and self.is_tracking:
                # Get the current tracker position
                tracker_rect = self.tracker.get_position()
                tracker_rect = (int(tracker_rect.left()), int(tracker_rect.top()),
                                int(tracker_rect.right()), int(tracker_rect.bottom()))

                # Check if the tracker rectangle is inside the body rectangle or if they intersect significantly
                if self.rectangles_overlap(tracker_rect, body_rectangle):
                    self.tracker = dlib.correlation_tracker()
                    self.tracker.start_track(
                        rgb_frame, dlib.rectangle(*body_rectangle))
                else:
                    # Continue to update the tracker
                    self.tracker.update(rgb_frame)

        if self.is_tracking and self.tracked_name:
            cv2.putText(frame, f"TRACKING {self.tracked_name.upper()}",
                        (20, 52), constants.FONT, 0.7, (0, 0, 255), 2)

            # Draw the dlib tracker rectangle in pink
            if self.tracker:
                tracker_rect = self.tracker.get_position()
                cv2.rectangle(frame, (int(tracker_rect.left()), int(tracker_rect.top())),
                              (int(tracker_rect.right()), int(tracker_rect.bottom())), (255, 0, 255), 2)

                centroid_x = (tracker_rect.left() + tracker_rect.right()) / 2
                centroid_y = (tracker_rect.top() + tracker_rect.bottom()) / 2 - (
                    tracker_rect.bottom() - tracker_rect.top()) / 4

                # Draw the centroid
                cv2.circle(frame, (int(centroid_x), int(
                    centroid_y)), 5, (0, 0, 255), -1)

            frame_center_x = frame.shape[1] // 2
            frame_center_y = frame.shape[0] // 2

            delta_x = 95  # Delta for left and right
            delta_y = 60  # Delta for up and down

            # Safe Zone
            cv2.ellipse(frame, (frame_center_x, frame_center_y), (delta_x, delta_y),
                        0, 0, 360, (0, 255, 0), 1)

            # Draw the center of the frame
            cv2.circle(frame, (frame_center_x, frame_center_y),
                       5, (255, 0, 0), -1)

            if self.ptz_controller is not None and centroid_x is not None:
                self.move_ptz(centroid_x, centroid_y, frame_center_x, frame_center_y,
                              delta_x, delta_y, frame.shape[1], frame.shape[0])

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

    def move_ptz(self, centroid_x, centroid_y, frame_center_x, frame_center_y, delta_x, delta_y, frame_width,
                 frame_height):
        """
        Uses Dlib Object Tracking to set and update the currently tracked person
        Then if a PTZ camera is associated, it should move the camera in any direction automatically
        :return:
        """

        # Calculate the distance of the centroid from the center along X and Y axes
        distance_x = abs(centroid_x - frame_center_x)
        distance_y = abs(centroid_y - frame_center_y)

        # Calculate the maximum possible distance from the center to a corner
        max_distance = math.sqrt(
            (frame_width / 2) ** 2 + (frame_height / 2) ** 2)

        # Calculate the speed
        speed_x, speed_y = self.calculate_speed(distance_x, distance_y, max_distance)

        # Set speed to 0 if the object is within the delta range
        if abs(centroid_x - frame_center_x) <= delta_x:
            speed_x = 0
        if abs(centroid_y - frame_center_y) <= delta_y:
            speed_y = 0

        self.ptz_control(centroid_x, centroid_y, speed_x, speed_y,
                         frame_center_x, frame_center_y, delta_x, delta_y)

        return

    def calculate_speed(self, distance_x, distance_y, max_distance):
        # Normalize the distances
        normalized_distance_x = distance_x / max_distance
        normalized_distance_y = distance_y / max_distance

        # Apply easing function
        # This is a simple quadratic easing function (ease in and out)
        eased_distance_x = normalized_distance_x * normalized_distance_x * (3 - 2 * normalized_distance_x)
        eased_distance_y = normalized_distance_y * normalized_distance_y * (3 - 2 * normalized_distance_y)

        # Scale to desired range of speeds
        if self.ptz_is_usb:
            speed_x = 2 + eased_distance_x * (7 - 2)
            speed_y = 2 + eased_distance_y * (7 - 2)
        else:
            # Adjust the speed calculation for NDI camera
            speed_x = 0.03 + eased_distance_x * (0.2 - 0.03)
            speed_y = 0.03 + eased_distance_y * (0.13 - 0.03)

        return round(speed_x, 2), round(speed_y, 2)

    def ptz_control(self, centerX, centerY, speed_x, speed_y, frame_center_x, frame_center_y, delta_x, delta_y):
        # Define the directions
        directions = [
            ("left_up", centerX < frame_center_x -
             delta_x and centerY < frame_center_y - delta_y),
            ("left_down", centerX < frame_center_x -
             delta_x and centerY > frame_center_y + delta_y),
            ("left", centerX < frame_center_x - delta_x),
            ("right_up", centerX > frame_center_x +
             delta_x and centerY < frame_center_y - delta_y),
            ("right_down", centerX > frame_center_x +
             delta_x and centerY > frame_center_y + delta_y),
            ("right", centerX > frame_center_x + delta_x),
            ("up", centerY < frame_center_y - delta_y),
            ("down", centerY > frame_center_y + delta_y),
            ("stop", abs(centerX - frame_center_x) <=
             delta_x and abs(centerY - frame_center_y) <= delta_y)
        ]

        for direction, condition in directions:
            if condition:
                if self.ptz_is_usb:
                    if direction == "stop":
                        self.ptz_controller.move_stop()
                    else:
                        getattr(self.ptz_controller,
                                f"move_{direction}_track")(speed_x)
                else:
                    if direction == "stop":
                        ndi.recv_ptz_pan_tilt_speed(
                            instance=self.ptz_controller, pan_speed=0, tilt_speed=0)
                    else:
                        # Determine the direction of movement for NDI camera
                        pan_speed = speed_x if "left" in direction else -speed_x
                        tilt_speed = speed_y if "up" in direction else -speed_y
                        ndi.recv_ptz_pan_tilt_speed(
                            instance=self.ptz_controller, pan_speed=pan_speed, tilt_speed=tilt_speed)
                self.last_request = direction
                break

    def closeEvent(self, event):
        """
        On event call, stop all the related threads.
        :param event:
        """
        if self.ptz_controller is not None:
            if self.ptz_is_usb:
                self.ptz_controller.move_stop()
            else:
                self.ptz_controller.pantilt(pan_speed=0, tilt_speed=0)
                self.ptz_controller.close_connection()
        self.camera_stream_process.terminate()
        self.facial_recognition_process.terminate()
        self.deleteLater()
        event.accept()
