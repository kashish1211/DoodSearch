import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
color = (0,255,255)

xp, yp = (0,0)
imgCanvas = np.zeros((480,640,3),np.uint8)
cv2.namedWindow('BGR',cv2.WINDOW_NORMAL)
cv2.resizeWindow('BGR', 720,720)




while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if (len(lmList)!=0):

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        

        if fingers[1] and fingers[2] == False:
            color = (255,255,255)
            cv2.circle(img, (x1,y1),5,color,cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1,y1
            
            cv2.line(img, (xp,yp),(x1,y1),color,5)
            cv2.line(imgCanvas, (xp,yp),(x1,y1),color,5)

            xp,yp = x1,y1

        if sum(fingers[1:5]) == 4 and not fingers[0]:
            print('Capture image')
            img_name = "opencv_frame.png"
            image_resized = "resized.png"
            imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
            _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
            imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
            cv2.imwrite(img_name, imgInv)
            cv2.resize(imgInv, (96,96))
            cv2.imwrite(image_resized,imgInv)

        if sum(fingers) == 5:
            print('Erase image')
            brushThickness = 30
            color = (0,0,0)
            cv2.circle(img, (x1,y1),brushThickness,color,cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1,y1
            
            cv2.line(img, (xp,yp),(x1,y1),color,brushThickness)
            cv2.line(imgCanvas, (xp,yp),(x1,y1),color,brushThickness)

            xp,yp = x1,y1



    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)
   
    cv2.imshow('BGR',img )  

    key = cv2.waitKey(10)
    if key == 27:
        break


cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
