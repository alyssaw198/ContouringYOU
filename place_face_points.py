import os
import cv2
import dlib
import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot
import matplotlib.colors as mcolors
import random
import pandas as pd

#uses 68 points
def get_face_points(img):
    '''
    image --> image
    Uses dlib to plot points along the bottom of the face and crops image such that the face is in the middle of the image
    '''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img)
    image_height, image_width, _ = img.shape

    #saves the x and y coordinates for the points that line the bottom of the face
    face_points = [[],[]]

    #iterates through the face and adds points to the edges
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        
        #marks the face with the points
        landmarks = predictor(gray_img,face)
        for n in range(17):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            face_points[0].append(x)
            face_points[1].append(y)
            #cv2.circle(img,(x,y),5,(255,255,255),cv2.FILLED)
    #crops the image so that the face is in the middle of the face
    cropped_image = img[:face_points[1][8]+10, face_points[0][0]-10:face_points[0][16]+10]
    #cv2.imshow("Cropped", cropped_image)
    #key = cv2.waitKey(0)
    return cropped_image


def get_face_points2(img):
    '''
    img --> list
    uses mediapipe to plot 468 points on the face, outlining features
    '''
    #wanted width and height of the image
    GLOBAL_WIDTH = 500
    GLOBAL_HEIGHT = 600
    mp_face_mesh = mp.solutions.face_mesh
    faceMesh = mp_face_mesh.FaceMesh()
    mp_draw = mp.solutions.drawing_utils

    frame = img
    frame = cv2.flip(frame, 1)
    results = faceMesh.process(frame)

    face_points = [[],[]]

    #find the scale to scale each image 
    image_height, image_width, _ = frame.shape
    height_scale = GLOBAL_HEIGHT/image_height
    width_scale = GLOBAL_WIDTH/image_width

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_FACE_OVAL, mp_draw.DrawingSpec((0,255,0), 1, 1))

            for id,lm in enumerate(face_landmarks.landmark):
                ih, iw, ic = frame.shape
                x,y = int(lm.x*iw), int(lm.y*ih)
                face_points[0].append(x)
                face_points[1].append(y)
                cv2.putText(frame, str(id), (x,y), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 255, 0), 1)
    
    #print(face_points)
    forehead_points = [face_points[0][10], face_points[1][10]]
    cropped_image = img[forehead_points[1]+10:, :]
    #cv2.imshow("Frame", cropped_image)
    #key = cv2.waitKey(0)
    
    #rescale each image to the desired size 
    new_x_points = [i * (width_scale) for i in face_points[0]]
    new_y_points = [i * (height_scale) for i in face_points[1]]
    face_points[0] = new_x_points
    face_points[1] = new_y_points
    return face_points


def all_face_points():     
    #get the path for current directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #specify the folder name for the images
    img_dir = os.path.join(BASE_DIR, "faces")

    #create a list to save all the image paths
    img_paths = []

    #get each image path and save into the list
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpeg") or file.endswith("jpg"):
                path = os.path.join(root, file)
                category = os.path.basename(os.path.dirname(path))
                img_paths.append(path)

    #create a dictionary to order the points on the images and the face shape it falls under
    points = {}

    #populate the dictionary
    for face_image in img_paths:
        img = cv2.imread(face_image, cv2.IMREAD_COLOR)
        category = os.path.basename(os.path.dirname(face_image))
        if category not in points:
            points[category] = [get_face_points2(get_face_points(img))]
        else:
            points[category].append(get_face_points2(get_face_points(img)))
    return points


def run():
    all_faces = all_face_points()

    def random_color():
        '''
        None --> Str
        '''
        color = random.choice(list(mcolors.CSS4_COLORS.keys()))
        return color

    face_average_points = {}

    #iterate through each face type and compute the average points for each plotted coordinate
    for face_type in all_faces:
        pyplot.xlim(0, 500)
        pyplot.ylim(150, 600)
        all_x = []
        all_y = []
        for face in all_faces[face_type]:
            all_x.append(face[0])
            all_y.append(face[1])
        df_x = pd.DataFrame(all_x)
        df_y = pd.DataFrame(all_y)
        df_x = df_x.mean()
        df_y = df_y.mean()
        x_avg_list = df_x.values.tolist()
        y_avg_list = df_y.values.tolist()
        face_average_points[face_type] = [x_avg_list, y_avg_list]
        #print(face_type)
        #pyplot.scatter(x_avg_list, y_avg_list, c="red", s=0.7)
        #pyplot.show()
    return face_average_points

if __name__ == "__main__":
    run()