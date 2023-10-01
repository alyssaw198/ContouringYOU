import numpy as np
import cv2
import mediapipe as mp

webcam = cv2.VideoCapture(0)

def rescaleFrame(frame, scale=0.75):
    #images, video, live video works
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

#flip = cv.flip(img, 1)

mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh()


while True:
    isTrue, frame = webcam.read()
    frame = cv2.flip(frame, 1)



    #frame_resized = rescaleFrame(frame, scale=0.5)
    cv2.imshow('webcam', frame)
    
    #cv2.imshow('Video Resized', frame_resized)

    key = cv2.waitKey(1)
    if key == 27:
        break

webcam.release()
cv2.destroyAllWindows()

cv2.waitKey(0)