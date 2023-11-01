from libraries.visca import camera
import threading


class ViscaPTZ:
    """
    Object Class to control all aspects of USB VISCA PTZ cameras
    """
    def __init__(self, device_id):
        if device_id != "":
            self.id = device_id
            self.visca_ptz = camera.D100(device_id)
            self.visca_ptz.init()
            print("Camera Initialized")

    def move_left_track(self, speed=7):
        """
        Continuous Movement to the left without stopping
        """
        try:
            self.visca_ptz.left(int(speed))
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_right_track(self, speed=7):
        """
        Continuous Movement to the right without stopping
        """
        try:
            self.visca_ptz.right(int(speed))
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_up_track(self, speed=7):
        """
        Continuous Movement to the up without stopping
        """
        try:
            self.visca_ptz.up(speed)
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_down_track(self, speed=7):
        """
        Continuous Movement to the down without stopping
        """
        try:
            self.visca_ptz.down(speed)
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_left_up_track(self, speed=5):
        """
        Continuous Movement to the left and up without stopping
        """
        try:
            self.visca_ptz.left_up(int(abs(speed)), int((abs(speed))))
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_right_up_track(self, speed=5):
        """
        Continuous Movement to the right and up without stopping
        """
        try:
            self.visca_ptz.right_up(int(abs(speed)), int((abs(speed))))
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_left_down_track(self, speed=5):
        """
        Continuous Movement to the left and down without stopping
        """
        try:
            self.visca_ptz.left_down(int(abs(speed)), int((abs(speed))))
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_right_down_track(self, speed=5):
        """
        Continuous Movement to the right and down without stopping
        """
        try:
            self.visca_ptz.right_down(int(abs(speed)), int((abs(speed))))
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_up(self):
        """
        Moves Camera up and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.up(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_left(self):
        """
        Moves Camera left and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.left(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_right(self):
        """
        Moves Camera right and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.right(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_down(self):
        """
        Moves Camera down and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.down(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_left_up(self):
        """
        Moves Camera left and up and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.left_up(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_right_up(self):
        """
        Moves Camera right and up and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.right_up(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_left_down(self):
        """
        Moves Camera left and down and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.left_down(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_right_down(self):
        """
        Moves Camera right and down and pauses for 0.4 seconds then stops the camera
        """
        try:
            self.visca_ptz.right_down(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_home(self):
        """
        Moves Camera to Home position and pauses for 3 seconds then stops the camera
        """
        try:
            self.visca_ptz.home()
            S = threading.Timer(3, self.move_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def move_stop(self):
        """
        Stops Camera Movement
        """
        try:
            self.visca_ptz.stop()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def menu(self):
        """
        Shows Camera Menu
        """
        try:
            self.visca_ptz.menu()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def zoom_in(self):
        """
        Zooms In Camera and pauses for 0.4 seconds then stops zoom in on camera
        """
        try:
            self.visca_ptz.zoom_in()
            S = threading.Timer(0.5, self.zoom_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def zoom_out(self):
        """
        Zooms Out Camera and pauses for 0.4 seconds then stops zoom out on camera
        """
        try:
            self.visca_ptz.zoom_out()
            S = threading.Timer(0.5, self.zoom_stop)
            S.start()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def zoom_stop(self):
        """
        Stops Camera Zooming
        """
        try:
            self.visca_ptz.zoom_stop()
        except Exception as e:
            print(f"Please initialize a camera {e}")

    def reset(self):
        """
        Reset PTZ, basically a quick restart without unplugging
        """
        try:
            self.visca_ptz.reset()
        except Exception as e:
            print(f"Please initialize a camera {e}")
