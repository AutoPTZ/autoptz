import cv2

cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
    print("Could not open video device")

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT,800)

while(True):    
    # Capture frame-by-frame    
    ret, frame = cap.read()    
    
    # Display the resulting frame    
    # cv2.imshow('preview',frame)    
    
    #gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #cv2.imshow('frame', gray)

    cv2.imshow('frame', frame)
    
    #Waits for a user input to quit the application    
    if cv2.waitKey(1) & 0xFF == ord('q'):    
        break
    
cap.release()
cv2.destroyAllWindows()