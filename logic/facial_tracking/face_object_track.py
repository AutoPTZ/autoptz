import cv2
import imutils

x = None
w = None
y = None
h = None

img = None
gray = None

enable_motion = None
track_started = None
track_name = None
tracker = cv2.TrackerCSRT_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# names related to ids: example ==> Steve: id=1 | try moving to trainer/labels.txt
labels_file = open("./trainer/labels.txt", "r")
names = labels_file.read().splitlines()
labels_file.close()


def click(event, x_pos, y_pos, flags, param):
    global enable_motion
    if event == cv2.EVENT_LBUTTONDOWN:
        if (x_pos > x) & (y_pos > y) * (x_pos < x + w) & (y_pos < y + h):
            enable_motion = not enable_motion


def face_object_track():
    recognizer.read('./trainer/trainer.yml')
    print("\n [INFO] Opening Advanced Recognition Software")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml");

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    global x
    global w
    global y
    global h
    global enable_motion
    global track_started
    global track_name
    global gray

    while True:
        timer = cv2.getTickCount()
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(int(minW), int(minH)))
        for (x_face, y_face, w_face, h_face) in faces:
            x = x_face
            w = w_face
            y = y_face
            h = h_face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y_face:y_face + h, x:x + w])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(frame, str(id), (x + 5, y_face - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        if enable_motion:
            cv2.putText(frame, "Tracking Enabled", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not track_started:
                tracker.init(frame, [x, y, w, h])
                print("Tracking Started @: " + str(x) + ' ' + str(y) + ' ' + str(w) + ' ' + str(h))
                track_name = id
                track_started = True
            success, bbox = tracker.update(frame)
            if len(faces) == 0:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), (255, 0, 255), 3, 1)
            else:
                if id == track_name:
                    tracker.init(frame, [x, y, w, h])
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
                else:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])), (255, 0, 255), 3, 1)
        else:
            cv2.putText(frame, "Tracking Disabled", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            track_started = False

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Advanced Recognition Software', frame)
        cv2.setMouseCallback("Advanced Recognition Software", click)

        key = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if key == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
