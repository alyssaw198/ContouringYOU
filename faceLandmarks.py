import cv2
import dlib 
import numpy as np 
from matplotlib import pyplot
from find_images import all_face_points
import matplotlib.colors as mcolors
import random

cap = cv2.VideoCapture(0) # open the video camera (default camera)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

points = []

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
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
        
        #Adds x y coordinates to a list
        points = list(range(17))
        x_points = list(range(17))
        y_points = list(range(17))
        
        for n in range(17):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            points[n] = [x,y]
            x_points[n] = x
            y_points[n] = y
            #if n == 8:
                #cv2.circle(frame, (x,y), 3, (255, 0, 0), -1)
            cv2.circle(frame, (x,y), 3, (255, 0, 0), -1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    #27 is the esc key
    if key == 27:
        break


def random_color():
    '''
    None --> Str
    '''
    color = random.choice(list(mcolors.CSS4_COLORS.keys()))
    return color

get_all_face_points = all_face_points()

for faces in get_all_face_points:
    for face in get_all_face_points[faces]:
        print(face[0])
        print(face[1])
        pyplot.scatter(face[0], face[1], c=random_color())

pyplot.scatter(x_points, y_points, c="red")
pyplot.show()
