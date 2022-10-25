import cv2
import dlib
import imutils

x = None
w = None
y = None
h = None

img = None
gray = None

enable_motion = None
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml");
# names related to ids: example ==> Steve: id=1 | try moving to trainer/labels.txt
labels_file = open("./trainer/labels.txt", "r")
names = labels_file.read().splitlines()
labels_file.close()
tracker = None


def click(event, x_pos, y_pos, flags, param):
    global enable_motion
    global x
    global y
    global w
    global h
    if event == cv2.EVENT_LBUTTONDOWN:
        if (x_pos > x) & (y_pos > y) & (x_pos < w) & (y_pos < h):
            enable_motion = not enable_motion


def face_object_track_dlib():
    recognizer.read('./trainer/trainer.yml')
    print("\n [INFO] Opening Advanced Recognition Software")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(2)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 3)

    # Define min window size to be recognized as a face
    minW = int(0.1 * cam.get(3))
    minH = int(0.1 * cam.get(4))

    global x
    global w
    global y
    global h
    global enable_motion
    global gray
    global tracker
    track_started = False
    enable_motion = False
    tracked_name = ""

    while True:
        timer = cv2.getTickCount()
        ret, orgFrame = cam.read()
        frame = imutils.resize(orgFrame, width=600)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(minW, minH))

        for (x_face, y_face, w_face, h_face) in faces:
            x = x_face
            w = x + w_face
            y = y_face
            h = y + h_face
            cv2.rectangle(frame, (x_face, y_face), (x_face + w_face, y_face + h_face), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y_face:y_face + h_face, x_face:x_face + w_face])
            # Check if confidence is less them 100 ==> "0" is perfect match
            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))
            cv2.putText(frame, str(id), (x_face + 5, y_face - 5), font, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, str(confidence), (x_face + w_face - 50, y_face + h_face + 15), font, 0.5, (255, 255, 0), 1)

        if enable_motion:
            cv2.putText(frame, "Tracking Enabled", (75, 75), font, 0.7, (0, 0, 255), 2)
            if not track_started:
                tracked_name = id
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(x, y, w, h)
                tracker.start_track(rgbFrame, rect)
                track_started = True
            if len(faces) == 0 | int(id is not tracked_name):
                tracker.update(rgbFrame)
                pos = tracker.get_position()
                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())
                cv2.rectangle(frame, (int(startX), int(startY)), (int(endX), int(endY)), (255, 0, 255), 3, 1)
                cv2.putText(frame, "tracking", (int(startX), int(endY + 15)), font, 0.45, (0, 255, 0), 1)
            else:
                rect = dlib.rectangle(int(x), int(y), int(w), int(h))
                tracker.start_track(rgbFrame, rect)
                cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (255, 0, 255), 3, 1)
                cv2.putText(frame, "tracking", (int(x), int(h + 15)), font, 0.45, (0, 255, 0), 1)
        else:
            cv2.putText(frame, "Tracking Disabled", (75, 75), font, 0.7, (0, 0, 255), 2)
            tracker = None
            track_started = False
            enable_motion = False

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


# needed for detecting a person
# # initialize the list of class labels MobileNet SSD was trained to detect
# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
#            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
#            "sofa", "train", "tvmonitor"]
# # load serialized model
# net = cv2.dnn.readNetFromCaffe("prototxt.txt", "MobileNetSSD_deploy.caffemodel")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)


# # initialize the video stream, dlib correlation tracker, output video
# print("[INFO] starting video stream...")
# vs = cv2.VideoCapture(0)
# tracker = None
#
# # loop over frames from the video file stream
# while True:
#     timer = cv2.getTickCount()
#     # grab the next frame from the video file
#     (grabbed, frame) = vs.read()
#
#     # resize the frame for faster processing and then convert the
#     # frame from BGR to RGB ordering (dlib needs RGB ordering)
#     frame = imutils.resize(frame, width=600)
#     rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     # if our correlation object tracker is None we first need to
#     # apply an object detector to seed the tracker with something
#     # to actually track
#     if tracker is None:
#         # grab the frame dimensions and convert the frame to a blob
#         (h, w) = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
#         # pass the blob through the network and obtain the detections
#         # and predictions
#         net.setInput(blob)
#         detections = net.forward()
#
#         # ensure at least one detection is made
#         if len(detections) > 0:
#             # find the index of the detection with the largest
#             # probability -- out of convenience we are only going
#             # to track the first object we find with the largest
#             # probability; future examples will demonstrate how to
#             # detect and extract *specific* objects
#             i = np.argmax(detections[0, 0, :, 2])
#             # grab the probability associated with the object along
#             # with its class label
#             conf = detections[0, 0, i, 2]
#             label = CLASSES[int(detections[0, 0, i, 1])]
#
#             # filter out weak detections by requiring a minimum
#             # confidence
#             if conf > .5 and label == "person":
#                 # compute the (x, y)-coordinates of the bounding box
#                 # for the object
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#                 # construct a dlib rectangle object from the bounding
#                 # box coordinates and then start the dlib correlation
#                 # tracker
#                 tracker = dlib.correlation_tracker()
#                 rect = dlib.rectangle(startX, startY, endX, endY)
#                 tracker.start_track(rgbFrame, rect)
#                 # draw the bounding box and text for the object
#                 cv2.rectangle(frame, (startX, startY), (endX, endY),
#                               (0, 255, 0), 2)
#                 cv2.putText(frame, label, (startX, startY - 15),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
#
#     # otherwise, we've already performed detection so let's track
#     # the object
#     else:
#         # update the tracker and grab the position of the tracked
#         # object
#         tracker.update(rgbFrame)
#         pos = tracker.get_position()
#         # unpack the position object
#         startX = int(pos.left())
#         startY = int(pos.top())
#         endX = int(pos.right())
#         endY = int(pos.bottom())
#         # draw the bounding box from the correlation object tracker
#         cv2.rectangle(frame, (startX, startY), (endX, endY),
#                       (0, 255, 0), 2)
#         cv2.putText(frame, label, (startX, startY - 15),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
#     # show the output frame
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
#     cv2.putText(frame, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.imshow("Frame", frame)
#     key = cv2.waitKey(1) & 0xFF
#     # if the `q` key was pressed, break from the loop
#     if key == ord("q"):
#         break
# # do a bit of cleanup
# cv2.destroyAllWindows()
# vs.release()
