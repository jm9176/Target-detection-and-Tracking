import cv2
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# Thresholding value for green
low = (90, 100, 100)
high = (120, 255, 255)

# Structuring elements for the morphological operation
kernel1 = np.ones((5,5), np.uint8)                # Kernel for erosion
kernel2 = np.ones((17,17), np.uint8)              # Kernel for dilation

cv2.namedWindow("camera")
cv2.namedWindow("rect")
cv2.namedWindow("ROI")

pt = (100,100)

while True:
    ret ,frame=cap.read()

    # Colorspace conversion
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    rect=hsv[pt[0]:pt[0]+300,pt[1]:pt[1]+300]

    # Applying the mask
    thresh = cv2.inRange(rect, low, high)

    # Applying teh morphological operations
    thresh = cv2.erode(thresh, kernel1, iterations = 2)
    thresh = cv2.dilate(thresh, kernel2, iterations = 2)

    #cv2.rectangle(frame,pt,(pt[0]+300,pt[1]+300),(0,0,255),2)

    # Display results
    cv2.imshow("camera", frame)
    cv2.imshow("rect", rect)
    cv2.imshow("ROI", thresh)

    cv2.waitKey(1)
    
cap.release()
cv2.destroyAllWindows()
