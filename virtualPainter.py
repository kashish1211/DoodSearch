from sys import flags
import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
from tkinter import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model

global flag
flag = 0

def open_camera():
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
        # print(img.shape)

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
                

            elif sum(fingers[1:4]) == 3 and not fingers[0] and not fingers[4]:
                brushThickness = 30
                color = (0,0,0)
                cv2.circle(img, (x1,y1),brushThickness,color,cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1,y1
                
                cv2.line(img, (xp,yp),(x1,y1),color,brushThickness)
                cv2.line(imgCanvas, (xp,yp),(x1,y1),color,brushThickness)

                xp,yp = x1,y1
                

            elif sum(fingers) == 5:
                # print('Capture image')
                img_name = "opencv_frame.png"    
                imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
                _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
                imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)            
                cv2.imwrite(img_name, imgInv)
                global flag
                flag = 1
                # time.sleep(5)
                break
            
            else:
                xp = 0
                yp = 0
                

                

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

open_camera()

if flag == 1:
    image = cv2.imread('opencv_frame.png')
    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    window_name = 'image'
    coords = cv2.findNonZero(image) # Find all non-zero points (text)
    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    image = image[y-10:y+h+10, x-10:x+w+10] 
    
    cv2.imshow(window_name, image)
    model = load_model('model.h5')
    new_classes = ['airplane','ambulance','apple','axe','backpack','banana','baseball_bat', 'basket', 'bat', 'bathtub', 
               'bed','bench', 'bicycle', 'binoculars', 'bird', 'birthday_cake','book', 'bowtie','bridge', 'broom', 'bucket',
               'bus','butterfly', 'cactus','calculator','camera', 'candle','car', 'carrot', 'cat', 'ceiling_fan', 'cell_phone', 'chair',
               'clock', 'cloud', 'coffee_cup', 'compass', 'computer', 'cookie','crab','crocodile', 'crown','cup', 'dog','donut', 'door', 'drums', 'duck', 'dumbbell', 'ear','elephant', 'envelope','eye', 'eyeglasses', 'face', 'fence', 'fire_hydrant', 'fireplace',
              'fish', 'flashlight', 'flower', 'foot', 'fork','giraffe','golf_club', 'grapes','guitar', 'hamburger', 'hammer', 'hand','hat','headphones','helicopter','hockey_stick','hot_air_balloon','hourglass', 'house',
              'ice_cream','key' ,'knife', 'ladder','leaf','light_bulb','lightning','lipstick','lollipop','matches','mountain',
              'mushroom', 'octopus','palm_tree','pants','passport', 'peanut','peas', 'pencil','pineapple', 'pizza','potato','purse','radio','rainbow', 'remote_control', 'rhinoceros','scissors','see_saw',
              'shoe', 'shorts', 'shovel','skateboard','smiley_face', 'snail', 'snowman','spoon','star','strawberry','suitcase', 'sun','sword', 'syringe','telephone', 'television', 'tennis_racquet', 'The_Eiffel_Tower',
              'tooth', 'toothbrush','traffic_light', 'train', 'tree','t-shirt', 'umbrella','wheel', 'windmill', 'wine_bottle','wristwatch']
    im = cv2.resize(image,(28,28))
    print(im.shape)
    im = im.reshape(1, 28, 28, 1).astype('float32')
    im /= 255.0
    pred = model.predict(im)[0]
    ind = (-pred).argsort()[:5]
    latex = [new_classes[x] for x in ind]
    print(latex)

    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 

    #closing all open windows 
    cv2.destroyAllWindows() 
    # win= Tk()
    # win.geometry("700x350")
    # im = Image.fromarray(image)
    # imgtk = ImageTk.PhotoImage(image=im)

    # Label(win, image= imgtk).pack()
    # win.mainloop()
