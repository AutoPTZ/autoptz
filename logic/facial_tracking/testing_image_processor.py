class ImageProcessor(QThread):
    def __init__(self, stream, width, height):
        super().__init__()
        self.stream = stream
        self.width = width
        self.height = height
        self._run_flag = True

        # CameraWidget will access these two variables
        self.face_locations = None
        self.face_names = None

        # Variables for Adding Faces, Recognition, and Tracking
        self.count = 0
        self.add_name = None
        self.tracking = None

    def run(self):
        while self._run_flag:
            frame = self.stream.frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.add_name:
                self.add_face(frame=frame, gray_frame=gray_frame)

    def add_face(self, frame, gray_frame):
        minW = 0.1 * gray_frame.shape[1]
        minH = 0.1 * gray_frame.shape[0]

        faces = constants.FACE_CASCADE.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=10,
                                                        minSize=(int(minW), int(minH)))
        self.face_locations = []
        self.face_names = []
        for x, y, w, h in faces:
            self.count = self.count + 1
            name = constants.IMAGE_PATH + self.adding_to_name + '/' + str(self.count) + '.jpg'
            print("\n [INFO] Creating Images........." + name)
            cv2.imwrite(name, frame)
            self.face_names.append(name)
            self.face_locations = [int(x/2), int(y/2), int((x + w)/2), int((y + h)/2)]

        if self.count >= 10:  # Take 50 face sample and stop video
            self.add_name = None
            self.count = 0
            self.face_locations = None
            self.face_names = None

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
