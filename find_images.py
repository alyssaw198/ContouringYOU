import os
import cv2
import dlib


def get_face_points(img, path_to_image):
    '''
    str --> list
    Takes in a path to an image and then outputs the points plotted on the face of the image
    '''
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img)

    face_points = [[],[]]

    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        landmarks = predictor(gray_img,face)
        for n in range(17):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            face_points[0].append(x)
            face_points[1].append(y)
            cv2.circle(img,(x,y),5,(255,255,255),cv2.FILLED)
    #cv2.imshow("Portrait", img)
    #key = cv2.waitKey(0)
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
            points[category] = [get_face_points(img, face_image)]
        else:
            points[category].append(get_face_points(img, face_image))
    return points

