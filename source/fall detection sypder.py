# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 03:35:03 2021

@author: Admin
"""

from twilio.rest import Client
import cv2
import time

fitToEllipse = False
cap = cv2.VideoCapture(r'E:\SB\Fall Detection\Data Collection\demo1.mp4')
time.sleep(2)

fgbg = cv2.createBackgroundSubtractorMOG2()
j = 0

while(1):
    ret, frame = cap.read()
    
    #Convert each frame to gray scale and subtract the background
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        
        #Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
        
            # List to hold all areas
            areas = []

            for contour in contours:
                ar = cv2.contourArea(contour)
                areas.append(ar)
            
            max_area = max(areas, default = 0)

            max_area_index = areas.index(max_area)

            cnt = contours[max_area_index]

            M = cv2.moments(cnt)
            
            x, y, w, h = cv2.boundingRect(cnt)

            cv2.drawContours(fgmask, [cnt], 0, (255,255,255), 3, maxLevel = 0)
            
            if h < w:
                j += 1
                
            if j > 10:
                print("FALL")
                cv2.putText(fgmask, 'FALL', (x, y), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
                
                t_s="ACc51294ea27ad1464b3e7b6000742c46c"
                t_a = "92f88891df24eb7dacdbc983d6c1e07e"
                client = Client(t_s,t_a)
                TO = "+917801028373"
                FROM="+12067923295"
                client.messages.create(to=TO,from_=FROM,body="Your Kid has FALLEN DOWN......(OR).....Your Parent has FALLEN DOWN")
                time.sleep(2000)

            if h > w:
                j = 0 
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


            cv2.imshow('video', frame)
        
            if cv2.waitKey(33) == 27:
             break
    except Exception as e:
        break
cv2.destroyAllWindows()