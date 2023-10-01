import cv2
import dlib 
import numpy as np 
import mediapipe as mp 

cap = cv2.VideoCapture(0) # open the video camera (default camera)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def createBox(frame, points, scale=5, masked=False,cropped=True):

    if masked:
        mask = np.zeros_like(frame)
        mask = cv2.fillPoly(mask,[points], (255,255,255))
        frame = cv2.bitwise_and(frame,mask)
        # cv2.imshow("Mask", frame) # isolated cropped mask frame

    if cropped:
        # cropping live camera to cropped frame
        bbox = cv2.boundingRect(points)
        x,y,w,h = bbox
        frameCrop = frame[y:y+h,x:x+w]
        frameCrop = cv2.resize(frameCrop, (0,0), None, scale, scale)
        return frameCrop
    else:
        return mask
    


while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

    faces = detector(gray)
    for face in faces: # looping through face -> coodinates of where the face is
        # extract the coord points
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # draw a rectangle around face
        # cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 255), 2)

        landmarks = predictor(gray, face)
        points = list(range(68))
        
        # mark each dot on face 
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points[n] = [x,y]
            cv2.circle(frame, (x,y), 3, (255, 0, 0), -1)
            cv2.putText(frame, str(n), (x,y-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0,0,255, 1))

        points = np.array(points)
        #frameLeftEye = createBox(frame, points[36:42])
        #cv2.imshow('Lefteye', frameLeftEye)
 
        frameLips = createBox(frame, points[48:61],3,masked=True,cropped=False)
        frameColorLips = np.zeros_like(frameLips)
        frameColorLips[:] = 20,0,55 # coloured frame 
        frameColorLips = cv2.bitwise_and(frameLips, frameColorLips) # coloring the lips on the masked frame
        frameColorLips = cv2.GaussianBlur(frameColorLips, (7,7), 10) # blurring the edges to make it smooth
        frameColorLips = cv2.addWeighted(frame,1,frameColorLips,0.4,0)

        cv2.imshow('Lips', frameLips)
        cv2.imshow("Colored", frameColorLips)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break

