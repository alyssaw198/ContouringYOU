import cv2
import mediapipe as mp

# face mesh
mp_face_mesh = mp.solutions.face_mesh
faceMesh = mp_face_mesh.FaceMesh()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0) # open the video camera (default camera)


while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    results = faceMesh.process(frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            print()
            mp_draw.draw_landmarks(frame, face_landmarks, None, mp_draw.DrawingSpec((0,255,0), 1, 1))
            
            # 




    # print(results.multi_face_landmarks)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if key == 27:
        break
    
  
