import cv2
import time
import imutils
import numpy as np
# Get the webcam
cap = cv2.VideoCapture(0)
# Setup the width and the height (your cam might not support these settings)
width = 640
height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


# Time is just used to get the Frames Per Second (FPS)
last_time = time.time()
while True:
    # Step 1: Capture the frame
    _, frame = cap.read()
    # Step 2: Convert to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Step 3: Create a mask based on medium to high Saturation and Value
    # - Hue 8-10 is about orange, which we will use
    # - These values can be changed (the lower ones) to fit your environment
    mask = cv2.inRange(hsv, (0, 0, 0), (50, 100, 100))
    # Step 4: This dilates with two iterations (makes it more visible)
    thresh = cv2.dilate(mask, None, iterations=5)
    # Step 5: Finds contours and converts it to a list
    contour = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contour)
    # Step 6: Loops over all objects found

    # (Extra) Add a FPS label to image
    text = f"FPS: {int(1 / (time.time() - last_time))}"
    last_time = time.time()
    cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    # Step 12: Show the frame
    cv2.imshow("Webcam", frame)
    cv2.imshow("HSV", hsv)
    cv2.imshow("contours", thresh)
    print(contours)
    # If q is pressed terminate
    if cv2.waitKey(1) == ord('p'):
        break
# Release and destroy all windows
cap.release()
cv2.destroyAllWindows()
