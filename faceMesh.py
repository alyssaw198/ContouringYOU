import cv2
import mediapipe as mp
import numpy as np
from matplotlib import pyplot
import dlib
from find_images import get_face_points2, get_face_points, run

# face mesh
mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0) # open the video camera (default camera)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

face_data_points = run()
personal_points = get_face_points2(get_face_points(frame))

face_similarity = {}

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

print("This is your face shape: " + most_similar_shape)