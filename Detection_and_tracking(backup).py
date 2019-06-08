import cv2
from darkflow.net.build import TFNet
import numpy as np
from time import sleep
import math
import datetime
import serial
from picamera.array import PiRGBArray
from picamera import PiCamera

options = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 3250,
    'threshold': 0.2
}
ser = serial.Serial('/dev/ttyACM0', 57600)

tfnet = TFNet(options)
colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

file =open("data.txt","w")

# Initialise the camera setup
camera = PiCamera()
camera.resolution = (320, 320)
camera.saturation = 100
camera.brightness = 60

rawCapture = PiRGBArray(camera, size=(320, 320))

# allow the camera to warmup
time.sleep(5)

# Camera capture for the object detection
camera.capture(rawCapture, format='bgr', use_video_port = True)
image = rawCapture.array
cv2.imwrite('image.png', image)
cv2.waitKey(0)
camera.stop_preview()
camera.close()
frame = cv2.imread('image.png')

ret = 1
while True:
    if ret:
        results = tfnet.return_predict(frame)
        for color, result in zip(colors, results):
            tl = (result['topleft']['x'], result['topleft']['y'])
            br = (result['bottomright']['x'], result['bottomright']['y'])
            frame = cv2.rectangle(frame, tl, br, (0,0,255), 2)
            w = result['bottomright']['x'] - result['topleft']['x']
            h = result['bottomright']['y'] - result['topleft']['y']

            cv2.imwrite('Rect.png',frame)

        imCrop = frame[b:b+h, a:a+w]

        # Thresholding to locate the center
        hsv = cv2.cvtColor(imCrop, cv2.COLOR_BGR2HSV)

        # Masking for the center
        lower_red = np.array([30,110,110])
        upper_red = np.array([80,255,255])
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Defining the morphological operations
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(mask,kernel,iterations = 2)

        kernel = np.ones((21,21),np.uint8)
        dilate = cv2.dilate(erosion,kernel,iterations = 2)

        # Finding the contours
        M = cv2.moments(dilate)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cv2.circle(imCrop,(cx, cy), 2, (0,0,255), -1)
        cy_n = b+cy
        cx_n = a+cx

        print("serial initialised")
        sleep(5)

# Camera Initialisation for the video output
        cam = PiCamera()
        cam.resolution = (320,320)
        cam.saturation = 100
        cam.brightness = 60
        rawCapture = PiRGBArray(cam, size=(320,320))
        time.sleep(5)

        # Parameters for LKOF
        lk_params = dict( winSize = (15,15),
                          maxLevel = 4,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        cam.capture(rawCapture, format='bgr', use_video_port = True)
        frame1 = rawCapture.array
        old_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        rawCapture.truncate(0)

        global point, point_selected, old_points

        point = (cx_n, cy_n)
        point_selected = True
        old_points = np.array([[cx_n, cy_n]], dtype = np.float32)

        for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port = True):
            tm = datetime.datetime.utcnow().strftime("%H:%M:%S")
            frame1 = frame.array
            gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

            if point_selected is True:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
                old_gray = gray_frame.copy()
                old_points = new_points
                x, y = new_points.ravel()

            # Bounding box size adjustment
            if count<1:
                im2, contours, hierarchy = cv2.findContours(dilate, 1, 2)
                cnt = contours[0]
                area = cv2.contourArea(cnt)
                r = math.sqrt(area)
                count=count+1

            frame1 = cv2.rectangle(frame1, (int(x-(1*r)),int(y-(0.8*r))),(int(x+(1*r)),int(y+(0.8*r))), (0,0,255), 2)

            # Extracting the new Bounding box
            imCrop1 = frame1[int(y-(0.8*r)):int(y+(0.8*r)), int(x-(1*r)):int(x+(1*r))]
            hsv1 = cv2.cvtColor(imCrop1, cv2.COLOR_BGR2HSV)

            # Threshold value for red
            lower_red = np.array([0,100,100])
            upper_red = np.array([30,255,255])
            mask_red = cv2.inRange(hsv1, lower_red, upper_red)

            # Threshold value for green
            lower_green = np.array([30,100,100])
            upper_green = np.array([75,255,255])
            mask_green = cv2.inRange(hsv1, lower_green, upper_green)

            # Threshold value for blue
            lower_blue = np.array([73,100,100])
            upper_blue = np.array([120,255,255])
            mask_blue = cv2.inRange(hsv1, lower_blue, upper_blue)

            # Defining the morphological operations
            kernel = np.ones((5,5),np.uint8)
            #erosion_r = cv2.erode(mask_red,kernel,iterations = 2)
            erosion_g = cv2.erode(mask_green,kernel,iterations = 2)
            #erosion_b = cv2.erode(mask_blue,kernel,iterations = 2)

            kernel = np.ones((21,21),np.uint8)
            dilate_r = cv2.dilate(erosion_r,kernel,iterations = 2)
            dilate_g = cv2.dilate(erosion_g,kernel,iterations = 2)
            dilate_b = cv2.dilate(erosion_b,kernel,iterations = 2)

            # Centroids of all the three targets
            Mr = cv2.moments(dilate_r)
            Mg = cv2.moments(dilate_g)
            Mb = cv2.moments(dilate_b)

            cx_g = int(Mg['m10']/Mg['m00'])
            cy_g = int(Mg['m01']/Mg['m00'])
            cx_r = int(Mr['m10']/Mr['m00'])
            cy_r = int(Mr['m01']/Mr['m00'])
            cx_b = int(Mb['m10']/Mb['m00'])
            cy_b = int(Mb['m01']/Mb['m00'])

            im2, contours, hierarchy = cv2.findContours(dilate_g, 1, 2)
            cnt = contours[0]
            area = cv2.contourArea(cnt)
            r1 = math.sqrt(area)
            r=r1

            # Height and width of the bounding box
            w=1.6*r
            h=1*r

            # Map the RoI coordinates wrt original frame
            cx_g = x-(w/2)+cx_g
            cy_g = y-(h/2)+cy_g
            cx_r = x-(w/2)+cx_r
            cy_r = y-(h/2)+cy_r
            cx_b = x-(w/2)+cx_b
            cy_b = y-(h/2)+cy_b

           # file.write(str(x)+ ', ' + str(y) + ', ' + str(cx_g)+ ', ' + str(cy_g)  + ', ' + str(cx_r)+ ', ' + str(cy_r) + ', ' + str(cx_b)+ ', ' + str(cy_b) + ', ' + str(w) + ', ' + str(h) + ', ' + str(pwm) + ', ' + str(tm) + ', ' +'\n')

            cv2.imshow("Frame", frame1)
            rawCapture.truncate(0)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        file.close()
        cam1.release()
        cv2.destroyAllWindows()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
