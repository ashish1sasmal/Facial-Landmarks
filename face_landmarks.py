import cv2
import numpy as np
import dlib
import sys
from imutils import face_utils
import imutils

detect = dlib.get_frontal_face_detector()
pred = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

img = cv2.imread(sys.argv[1])
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

rects = detect(gray,2)


for (i,rect) in enumerate(rects):
    shape = pred(gray,rect)
    shape  = face_utils.shape_to_np(shape)

    (x,y,w,h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0), 2)

    cv2.putText(img, f"Face #{i+1}", (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x,y) in shape:
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

cv2.imshow("Output", img)
cv2.waitKey(0)
