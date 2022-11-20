import cv2
import shared.constants as constants
from threading import Thread


class ImageProcessor:
    def __init__(self, stream_thread, width=None, height=None):
        super().__init__()
        self.stream = stream_thread
        self.width = width
        self.height = height
        self._run_flag = True

        # CameraWidget will access these two variables
        self.face_locations = None
        self.face_names = None
        self.confidence_list = None

        # Variables for Adding Faces, Recognition, and Tracking
        self.count = 0
        self.add_name = None
        self.tracking = None

    def start(self):
        Thread(target=self.process, args=()).start()
        return self

    def process(self):
        while self._run_flag:
            frame = self.stream.cv_img
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.add_name:
                self.add_face(frame=frame, gray_frame=gray_frame)

    def add_face(self, frame, gray_frame):
        min_w = 0.1 * gray_frame.shape[1]
        min_h = 0.1 * gray_frame.shape[0]

        faces = constants.FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10,
                                                        minSize=(int(min_w), int(min_h)))
        self.face_locations = []
        self.face_names = []
        self.confidence_list = []
        for x, y, w, h in faces:
            self.count = self.count + 1
            location = constants.IMAGE_PATH + self.add_name + '/' + str(self.count) + '.jpg'
            print("\n [INFO] Creating Images at " + location)
            cv2.imwrite(location, frame)
            self.face_names.append(self.add_name)
            self.face_locations = [(int(x / 2), int(y / 2), int((x + w) / 2), int((y + h) / 2))]
            self.confidence_list.append(0)

        if self.count >= 10:  # Take 50 face sample and stop video
            self.add_name = None
            self.count = 0
            self.face_locations = None
            self.face_names = None

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
