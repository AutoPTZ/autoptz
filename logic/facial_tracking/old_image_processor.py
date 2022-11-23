class ImageProcessor(QtCore.QObject):
    START_TRAINER = QtCore.pyqtSignal()

    def __init__(self):
        super(ImageProcessor, self).__init__()

        # VISCA/ONVIF PTZ Control
        self.ptz_ready = None
        self.camera_control = None

    def track_face(self, frame, x, y, w, h):
        if self.camera_control is not None:
            if self.ptz_ready is None:
                # For VISCA PTZ
                if x > min_x and w < max_x and y > min_y and h < max_y:
                    self.camera_control.move_stop()
                if w > max_x:
                    self.camera_control.move_right_track()
                elif x < min_x:
                    self.camera_control.move_left_track()
                if h > max_y:
                    self.camera_control.move_down_track()
                elif y < min_y:
                    self.camera_control.move_up_track()
            else:
                # For ONVIF PTZ
                if x > min_x and w < max_x and y > min_y and h < max_y:
                    self.camera_control.stop_move()
                    # movementX = False
                    # faster_movement = False
                if w > max_x:
                    self.camera_control.continuous_move(0.05, 0, 0)
                    # movementX = False
                elif x < min_x:
                    self.camera_control.continuous_move(-0.05, 0, 0)
                    # movementX = False
                if h > min_y:
                    self.camera_control.continuous_move(0, -0.05, 0)
                    # movementY = False
                elif y < max_y:
                    self.camera_control.continuous_move(0, 0.05, 0)
                    # movementY = False
        return frame

    def set_ptz_ready(self, text):
        self.ptz_ready = text

    def get_ptz_ready(self):
        return self.ptz_ready

    def set_ptz_controller(self, control):
        self.camera_control = control
