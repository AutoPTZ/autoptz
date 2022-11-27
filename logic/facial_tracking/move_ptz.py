import time
from threading import Thread

from PySide6.QtCore import QThread


class MovePTZ(QThread):

    def __init__(self, ptz_controller, ptz_request_queue, lock, isVISCA=False):
        super(MovePTZ, self).__init__()
        self.ptz_control = ptz_controller
        self.queue = ptz_request_queue
        self.isVISCA = isVISCA
        self._run_flag = True
        self.lock = lock
        print("MovePTZ started")

    def run(self):
        while self._run_flag:
            self.lock.acquire(blocking=True)
            if self.queue.empty() is False:
                request = self.queue.get(timeout=0.3)
                if request == "stop":
                    self.ptz_control.pantilt(pan_speed=0, tilt_speed=0)
                if request == "left":
                    self.ptz_control.pantilt(pan_speed=1, tilt_speed=0)
                if request == "right":
                    self.ptz_control.pantilt(pan_speed=-1, tilt_speed=0)
                if request == "down":
                    self.ptz_control.pantilt(pan_speed=0, tilt_speed=1)
                if request == "up":
                    self.ptz_control.pantilt(pan_speed=0, tilt_speed=-1)
                print(request)
            self.lock.release()

    def stop(self):
        self._run_flag = False
        self.ptz_control.pantilt(pan_speed=0, tilt_speed=0)
        self.ptz_control.close_connection()
        self.wait()
