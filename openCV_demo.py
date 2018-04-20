import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)


while True:
    ret, img = cap.read()

    r = 640.0 / img.shape[1]
    dim = (640, int(img.shape[0] * r))

    # perform the actual resizing of the image and show it
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)


    cv2.imshow('camera', img)
    if cv2.waitKey(1) == 27:
            break  # esc to quit
    cv2.destroyAllWindows()
