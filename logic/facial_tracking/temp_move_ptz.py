import time
from threading import Thread


# Might delete depending on further testing of Network PTZ

class MovePTZ(Thread):

    def __init__(self, ptz_controller, ptz_request_queue, lock, isUSB=False):
        super(MovePTZ, self).__init__()
        self.ptz_control = ptz_controller
        self.queue = ptz_request_queue
        self.isUSB = isUSB
        self._run_flag = True
        self.lock = lock
        print("MovePTZ started")

    def run(self):
        while self._run_flag:
            self.lock.acquire(blocking=True)
            if self.queue.empty() is False:
                request = self.queue.get(timeout=None, block=True)
                if self.isUSB:
                    if request == "stop":
                        self.ptz_control.move_stop()
                    if request == "left":
                        self.ptz_control.move_left_track()
                    if request == "right":
                        self.ptz_control.move_right_track()
                    if request == "down":
                        self.ptz_control.move_down_track()
                    if request == "up":
                        self.ptz_control.move_up_track()
                else:
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
                time.sleep(0.001)
            self.lock.release()

    def stop(self):
        self._run_flag = False
        if self.isUSB:
            self.ptz_control.move_stop()
        else:
            self.ptz_control.pantilt(pan_speed=0, tilt_speed=0)
            self.ptz_control.close_connection()
