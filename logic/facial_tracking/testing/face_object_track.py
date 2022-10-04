import cv2


# def drawBox(img, bbox):
#     cap = cv2.VideoCapture(0)
#
#     tracker = cv2.TrackerCSRT_create()
#     success, img = cap.read()
#     bbox = cv2.selectROI("Tracking", img, False)
#     tracker.init(img, bbox)
#
# x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
# cv2.rectangle(img, (x, y), ((x + w), (y + h)), (255, 0, 255), 3, 1)
# cv2.putText(img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#
#
# while True:
#     timer = cv2.getTickCount()
#     success, img = cap.read()
#
#     success, bbox = tracker.update(img)
#
#     if success:
#         drawBox(img, bbox)
#     else:
#         cv2.putText(img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#
#     fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
#     cv2.putText(img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.imshow("Tracker", img)
#
#     if cv2.waitKey(1) & 0xff == ord('q'):
#         break


def temp_face_object_track():
    print("\n [INFO] Opening Advanced Recognition Software")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('../trainer/trainer.yml')
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml");

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['Lalo', 'Steve']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)

    # create x, y, w, h variables
    x_min = 0
    y_min = 0
    x_max = 0
    y_max = 0

    def select_face_event(event, x, y, flags, args):
        if event == cv2.EVENT_LBUTTONDOWN:
            if x > x_min & x < x_max & y > y_min & y < y_max:
                print("starting object tracker for " + str(id))

    # Mouse Event for Selecting Face Detected
    # def select_face_event(event, x, y, flags, args):
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         if x > x_min & x < x_max & y > y_min & y < y_max:
    #             print("in range to track")
    #             print(str(id))
    #             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 3, 1)

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            x_min = x
            y_min = y
            x_max = x + w
            y_max = y + h
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less them 100 ==> "0" is perfect match

            if confidence < 100:
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 255), 3, 1)
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        cv2.setMouseCallback('camera', select_face_event)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()


def main():
    temp_face_object_track()


if __name__ == '__main__':
    main()
