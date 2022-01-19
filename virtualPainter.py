import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm

folderPath = 'header'
# myList = os.listdir(folderPath)

header = cv2.imread(f'{folderPath}/topbar.jpg')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)
color = (0,255,0)
brushThickness = 15
xp, yp = (0,0)
imgCanvas = np.zeros((720,1280,3),np.uint8)
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if (len(lmList)!=0):
        # print(lmList)

        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()
        # print(fingers)
        
        if fingers[1] and fingers[2]:
            xp,yp = 0,0

            cv2.rectangle(img,(x1,y1-25),(x2,y2+25),color,cv2.FILLED)
            print('selection mode')
            if y1 < 137:
                if 88<x1<238:
                    color = (0,255,0)
                    brushThickness = 15   
                elif 326<x1<476:
                    color = (0,0,255)
                    brushThickness = 15   
                elif 564<x1<714:
                    color = (203,192,255)
                    brushThickness = 15   
                elif 802<x1<952:
                    color = (255,0,0)
                    brushThickness = 15                   
                elif 1040<x1<1190:
                    brushThickness = 30
                    color = (0,0,0)
                else:
                    print("blank")

        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1,y1),5,color,cv2.FILLED)
            print('Drawing mode')
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


    img[0:137,0:1280] = header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow('Image',img)    
    # cv2.imshow('Image canvas',imgCanvas)

    key = cv2.waitKey(10)
    if key == 27:
        break


cv2.destroyAllWindows() 
cv2.VideoCapture(0).release()
