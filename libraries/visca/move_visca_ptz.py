from libraries.visca import camera
import threading


class ViscaPTZ:

    def __init__(self, device_id):
        if device_id != "":
            self.id = device_id
            self.visca_ptz = camera.D100(device_id)
            self.visca_ptz.init()
            print("Camera Initialized")

    def move_left_track(self):
        try:
            self.visca_ptz.left(7)
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_right_track(self):
        try:
            self.visca_ptz.right(7)
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_up_track(self):
        try:
            self.visca_ptz.up(7)
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_down_track(self):
        try:
            self.visca_ptz.down(7)
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_up(self):
        try:
            self.visca_ptz.up(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_left(self):
        try:
            self.visca_ptz.left(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_right(self):
        try:
            self.visca_ptz.right(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_down(self):
        try:
            self.visca_ptz.down(5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_left_up(self):
        try:
            self.visca_ptz.left_up(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_right_up(self):
        try:
            self.visca_ptz.right_up(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_left_down(self):
        try:
            self.visca_ptz.left_down(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_right_down(self):
        try:
            self.visca_ptz.right_down(5, 5)
            S = threading.Timer(0.4, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_home(self):
        try:
            self.visca_ptz.home()
            S = threading.Timer(3, self.move_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def move_stop(self):
        try:
            self.visca_ptz.stop()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def menu(self):
        try:
            self.visca_ptz.menu()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def zoom_in(self):
        try:
            self.visca_ptz.zoom_in()
            S = threading.Timer(0.5, self.zoom_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def zoom_out(self):
        try:
            self.visca_ptz.zoom_out()
            S = threading.Timer(0.5, self.zoom_stop)
            S.start()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def zoom_stop(self):
        try:
            self.visca_ptz.zoom_stop()
        except Exception as e:
            print(e)
            print("Please initialize a camera")

    def reset(self):
        try:
            self.visca_ptz.reset()
        except Exception as e:
            print(e)
            print("Please initialize a camera")
