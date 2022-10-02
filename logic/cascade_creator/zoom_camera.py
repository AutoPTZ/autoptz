from time import time
import cv2


cap = cv2.VideoCapture(0)

loop_time = time()
scale = 10
while True:
    # get an updated image of the game
    ret, frame = cap.read()

    # get the webcam size
    height, width, channels = frame.shape

    # prepare the crop
    centerX, centerY = int(height / 2), int(width / 2)
    radiusX, radiusY = int(scale * height / 100), int(scale * width / 100)

    minX, maxX = centerX - radiusX, centerX + radiusX
    minY, maxY = centerY - radiusY, centerY + radiusY

    cropped = frame[minX:maxX, minY:maxY]
    resized_cropped = cv2.resize(cropped, (width, height))

    # display the images
    cv2.imshow('Capture Target', resized_cropped)

    # debug the loop rate
    #print('FPS {}'.format(1 / (time() - loop_time)))
    loop_time = time()

    # press 'q' with the output window focused to exit.
    # press 'f' to save screenshot as a positive image,
    # press 'd' to save as a negative image.
    # waits 1 ms every loop to process key presses
    key = cv2.waitKey(1)
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
    elif key == ord('f'):
        cv2.imwrite('positives/{}.jpg'.format(loop_time), resized_cropped)
    elif key == ord('d'):
        cv2.imwrite('negatives/{}.jpg'.format(loop_time), resized_cropped)
    elif key == ord('i'):
        print("plus")
        scale += 5  # +5
        print(scale)
    elif key == ord('o'):
        print("minus")
        scale = 5  # +5
        print(scale)

print('Done.')
