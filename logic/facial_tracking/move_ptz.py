from threading import Thread


class MovePTZ(Thread):

    def __init__(self, ptz_controller):
        super(MovePTZ, self).__init__()
        self.ptz_control = ptz_controller
        self.is_ready = False
        self._run_flag = True
        self.moving_right = False
        self.moving_left = False
        self.moving_up = False
        self.moving_down = False
        self.stop_moving = False

    def stop_moving(self):
        self.ptz_control.stop_move()

    def move_right(self):
        self.ptz_control.continuous_move(0.05, 0, 0)

    def move_left(self):
        self.ptz_control.continuous_move(-0.05, 0, 0)


    # def run(self):
    #     while self._run_flag:
    #         print("moving")
    #         self.ptz_control.continuous_move(0.05, 0, 0)
    #         if self.is_ready:
    #             self.is_ready = False
    #             if self.stop_moving:
    #                 self.ptz_control.stop_move()
    #                 self.stop_moving = False
    #             if self.moving_right:
    #                 self.ptz_control.continuous_move(0.05, 0, 0)
    #                 self.moving_right = False
    #             if self.moving_left:
    #                 self.ptz_control.continuous_move(-0.05, 0, 0)
    #                 self.moving_left = False
    #             # if h > min_y:
    #             #     self.ptz_control.continuous_move(0, -0.05, 0)
    #             #     # movementY = False
    #             # elif y < max_y:
    #             #     self.ptz_control.continuous_move(0, 0.05, 0)
    #             self.is_ready = True

    def stop(self):
        self._run_flag = False
