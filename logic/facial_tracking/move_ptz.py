import time
from threading import Thread

from PySide6.QtCore import QThread


class MovePTZ(Thread):

    def __init__(self, ptz_controller, lock):
        super(MovePTZ, self).__init__()
        self.ptz_control = ptz_controller
        self._run_flag = True
        self.lock = lock

        self._moving_left = False
        self._moving_right = False
        self._moving_up = False
        self._moving_down = False

        self.done_moving = True
        self._is_stopped = False

    def stop_moving(self):
        self.ptz_control.stop_move()
        self._is_stopped = False
        self.done_moving = True
        self._moving_left = False
        self._moving_right = False
        self._moving_up = False
        self._moving_down = False
        time.sleep(0.5)

    def move_right(self):
        self.ptz_control.continuous_move(0.05, 0, 0)
        self._moving_right = False
        time.sleep(0.1)

    def move_left(self):
        self.ptz_control.continuous_move(-0.05, 0, 0)
        self._moving_left = False
        time.sleep(0.1)

    def move_up(self):
        self.ptz_control.continuous_move(0, 0.05, 0)
        self._moving_up = False
        time.sleep(0.1)

    def move_down(self):
        self.ptz_control.continuous_move(0, -0.05, 0)
        self._moving_down = False
        time.sleep(0.1)

    def run(self):
        while self._run_flag:
            self.lock.acquire(blocking=True)
            # if self._is_stopped and self.done_moving:
            #     self.stop_moving()
            if self._moving_left:
                self.move_left()
            elif self._moving_right:
                self.move_right()
            # elif self._moving_up:
            #     self.move_up()
            # elif self.move_down():
            #     self.move_down()
            self.lock.release()


    def stop(self):
        self._run_flag = False
        # self.stop_moving()
        # self.wait()
