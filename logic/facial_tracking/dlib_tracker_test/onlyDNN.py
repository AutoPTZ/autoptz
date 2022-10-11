# How to run?: python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
# python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
import dlib as dlib
# import packages
import numpy as np
import imutils
import time
import cv2

from logic.facial_tracking.test_1.CentroidTracker import CentroidTracker

# Let's start by initialising the list of the 21 class labels MobileNet SSD was trained to.
# Each prediction composes of a boundary box and 21 scores for each class (one extra class for no object),
# and we pick the highest score as the class for the bounded object
CLASSES = ["aeroplane", "background", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

# Assigning random colors to each of the classes
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# COLORS: a list of 21 R,G,B values, like ['101.097383   172.34857188 111.84805346'] for each label
# length of COLORS = length of CLASSES = 21

# load our serialized model
# The model from Caffe: MobileNetSSD_deploy.prototxt.txt; MobileNetSSD_deploy.caffemodel;
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("prototxt.txt", "MobileNetSSD_deploy.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# print(net)
# <dnn_Net 0x128ce1310>

# initialize the video stream,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(0)
# warm up the camera for a couple of seconds
time.sleep(2.0)

while True:
    timer = cv2.getTickCount()
    # grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
    # vs is the VideoStream
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=500)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # grab the frame dimensions and convert it to a blob
    # First 2 values are the h and w of the frame. Here h = 225 and w = 400
    (h, w) = frame.shape[:2]
    # Resize each frame
    # resized_image = cv2.resize(frame, (300, 300))
    # Creating the blob
    # The function:
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    # image: the input image we want to preprocess before passing it through our deep neural network for classification
    # mean:
    # scalefactor: After we perform mean subtraction we can optionally scale our images by some factor. Default = 1.0
    # scalefactor  should be 1/sigma as we're actually multiplying the input channels (after mean subtraction) by scalefactor (Here 1/127.5)
    # swapRB : OpenCV assumes images are in BGR channel order; however, the 'mean' value assumes we are using RGB order.
    # To resolve this discrepancy we can swap the R and B channels in image  by setting this value to 'True'
    # By default OpenCV performs this channel swapping for us.

    blob = cv2.dnn.blobFromImage(frame, (1 / 127.5), (300, 300), 127.5, swapRB=True)
    # print(blob.shape) # (1, 3, 300, 300)
    # pass the blob through the network and obtain the predictions and predictions
    net.setInput(blob)  # net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    # Predictions:
    predictions = net.forward()
    # number of persons detected

    n_persons = 0
    boxes_person = []

    # loop over the predictions
    for i in np.arange(0, predictions.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        # predictions.shape[2] = 100 here
        confidence = predictions[0, 0, i, 2]
        # Filter out predictions lesser than the minimum confidence level
        # Here, we set the default confidence as 0.2. Anything lesser than 0.2 will be filtered
        if confidence > .5:
            # extract the index of the class label from the 'predictions'
            # idx is the index of the class label
            # E.g. for person, idx = 15, for chair, idx = 9, etc.
            idx = int(predictions[0, 0, i, 1])
            # then compute the (x, y)-coordinates of the bounding box for the object
            box = predictions[0, 0, i, 3:7] * np.array([w, h, w, h])
            # Example, box = [130.9669733   76.75442174 393.03834438 224.03566539]
            # Convert them to integers: 130 76 393 224
            (startX, startY, endX, endY) = box.astype("int")

            # draw the prediction on the frame
            # draw only for class "person"
            if CLASSES[idx] == "person":
                n_persons = n_persons + 1

                label = "{}: {:.2f}%".format(CLASSES[idx],
                                             confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            # b = sgbox(startX, startY, endX, endY)
            # boxes_person.append(list(b.exterior.coords))
            # Put a text outside the rectangular detection
            # Choose the font of your choice: FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX, FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC, etc.
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        # show the output frame
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # HOW TO STOP THE VIDEOSTREAM?
    # Using cv2.waitKey(1) & 0xFF

    # The waitKey(0) function returns -1 when no input is made
    # As soon an event occurs i.e. when a button is pressed, it returns a 32-bit integer
    # 0xFF represents 11111111, an 8 bit binary
    # since we only require 8 bits to represent a character we AND waitKey(0) to 0xFF, an integer below 255 is always obtained
    # ord(char) returns the ASCII value of the character which would be again maximum 255
    # by comparing the integer to the ord(char) value, we can check for a key pressed event and break the loop
    # ord("q") is 113. So once 'q' is pressed, we can write the code to break the loop
    # Case 1: When no button is pressed: cv2.waitKey(1) is -1; 0xFF = 255; So -1 & 255 gives 255
    # Case 2: When 'q' is pressed: ord("q") is 113; 0xFF = 255; So 113 & 255 gives 113

    # Explaining bitwise AND Operator ('&'):
    # The & operator yields the bitwise AND of its arguments
    # First you convert the numbers to binary and then do a bitwise AND operation
    # For example, (113 & 255):
    # Binary of 113: 01110001
    # Binary of 255: 11111111
    # 113 & 255 = 01110001 (From the left, 1&1 gives 1, 0&1 gives 0, 0&1 gives 0,... etc.)
    # 01110001 is the decimal for 113, which will be the output
    # So we will basically get the ord() of the key we press if we do a bitwise AND with 255.
    # ord() returns the unicode code point of the character. For e.g., ord('a') = 97; ord('q') = 113

    # Now, let's code this logic (just 3 lines, lol)
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' key to break the loop
    if key == ord("q"):
        break

# Destroy windows and cleanup
cv2.destroyAllWindows()
# Stop the video stream
vs.stop()

# In case you removed the opaque tape over your laptop cam, make sure you put them back on once finished ;)
# YAYYYYYYYYYY WE ARE DONE!
