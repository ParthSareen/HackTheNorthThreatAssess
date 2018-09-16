import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while (True):
    #capturing data frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor = 1.5, minNeighbors = 5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w] #(ycord_start-height, ycord_end-height)
        img_item = "my-image.png" #creates image for testing

        #recognize? deep learning model predict keras tensorflow pytorch scikit learn


        cv2.imwrite(img_item, roi_gray)
        print(x,y,w,h)

        color = (0, 0, 255) #BGR
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y),(end_cord_x, end_cord_y), color, stroke)

    #display frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#when proccesses are done executing, release VideoCapture
cap.release()
cv2.destroyAllWindows()
