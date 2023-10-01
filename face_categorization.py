import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot
import dlib
from place_face_points import get_face_points2, get_face_points, run
import training_data as td
import os
#import tkinter as tk
from tkinter import *


def get_face_shape():
    '''
    None --> str
    Accesses the user's face through a webcam to determine the correct face shape to assign the user
    '''
    # face mesh
    mp_face_mesh = mp.solutions.face_mesh
    faceMesh = mp_face_mesh.FaceMesh()
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0) # open the video camera (default camera)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    root = Tk()
    root.title("MESSAGE")
    root.eval('tk::PlaceWindow . center')
    w = 600 # Width 
    h = 300 # Height
    screen_width = root.winfo_screenwidth()  # Width of the screen
    screen_height = root.winfo_screenheight() # Height of the screen
    x = (screen_width/2) - (w/2)
    y = (screen_height/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    label = Label(root, text="HOW TO USE\n\nClick <Go to Analysis> to activate the camera.\n Make sure your face is centered in front of the camera \n and click ESC to continue once you are satisfied with the positioning.").pack()
    #root.after(20000, lambda: root.destroy())
    exit_button = Button(root, text="Go to Analysis", command=root.destroy)
    exit_button.pack(pady=20)
    root.mainloop()



    #access the webcame and get an image of the user's face
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    #cv2.destroyWindow("Frame")
        
    #get the data of the different face shapes
    face_data_points = td.training
    #get the data for the user's face 
    personal_points = get_face_points2(get_face_points(frame))

    face_similarity = {}

    #find the Euclidean distance to calculate the similarities between the user's the other face shapes
    for face_type in face_data_points:
        x1 = face_data_points[face_type][0]
        y1 = face_data_points[face_type][1]
        x2 = personal_points[0]
        y2 = personal_points[1]

        array_x1 = np.array(x1)
        array_y1 = np.array(y1)
        array_x2 = np.array(x2)
        array_y2 = np.array(y2)

        # Calculate the Euclidean distance between corresponding points in the two arrays
        distances = np.sqrt((array_x1 - array_x2)**2 + (array_y1 - array_y2)**2)

        # Calculate a similarity score based on the distances
        similarity_score = 1 / (1 + distances.sum())
        face_similarity[face_type] = similarity_score

    #smaller similarity (smaller Euclidean distance)= more similar
    smallest = float("inf")
    most_similar_shape = ""
    for face_type in face_similarity:
        if face_similarity[face_type] < smallest:
            smallest = face_similarity[face_type]
            most_similar_shape = face_type
    #root.destroy()
    cv2.destroyWindow("Frame")
    #create a pop up notification
    root = Tk()
    root.title("MESSAGE")
    root.eval('tk::PlaceWindow . center')
    w = 600 # Width 
    h = 300 # Height
    screen_width = root.winfo_screenwidth()  # Width of the screen
    screen_height = root.winfo_screenheight() # Height of the screen
    x = (screen_width/2) - (w/2)
    y = (screen_height/2) - (h/2)
    root.geometry('%dx%d+%d+%d' % (w, h, x, y))
    label = Label(root, text="This is your face shape:\n\n" + most_similar_shape).pack()
    #root.after(20000, lambda: root.destroy())
    exit_button = Button(root, text="View your makeup look", command=root.destroy)
    exit_button.pack(pady=20)
    root.mainloop()
    return most_similar_shape

if __name__ == "__main__":
    get_face_shape()
    