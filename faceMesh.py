import cv2
import mediapipe as mp
import numpy as np 
import draw as draw

def createBox(frame, points, scale=5, masked=False,cropped=True):
    
    if masked:
        mask = np.zeros_like(frame)
        mask = cv2.fillPoly(mask, [points], (255,255,255))
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

def diamondShape(frame, list1):

    # diamond left undereye
    list5 = []
    list5.append(list1[31])
    list5.append(list1[233])
    list5.append(list1[205])
    list5 = np.array(list5)

    # left cheek
    list6 = []
    list6.append(list1[57])
    list6.append(list1[212])
    list6.append(list1[214])
    list6.append(list1[192])
    list6.append(list1[213])
    list6.append(list1[147])
    list6.append(list1[187])
    list6.append(list1[207])
    list6.append(list1[216])
    list6 = np.array(list6)

    # chin
    list7 = []
    list7.append(list1[208])
    list7.append(list1[175])
    list7.append(list1[428])
    list7.append(list1[200])
    list7 = np.array(list7)

    # right cheek
    list8 = []
    list8.append(list1[287])
    list8.append(list1[432])
    list8.append(list1[434])
    list8.append(list1[416])
    list8.append(list1[433])
    list8.append(list1[376])
    list8.append(list1[411])
    list8.append(list1[427])
    list8.append(list1[436])
    list8 = np.array(list8)


    # right under eye
    list9 = []
    list9.append(list1[453])
    list9.append(list1[261])
    list9.append(list1[425])
    list9 = np.array(list9)

    frameForeHead = createBox(frame, list4, 3, masked=True, cropped=False)
    frameColorFH = np.zeros_like(frameForeHead)
    frameColorFH[:] = 226,226,255 # coloured frame 
    frameColorFH = cv2.bitwise_and(frameForeHead, frameColorFH) # coloring the lips on the masked frame
    frameColorFH = cv2.GaussianBlur(frameColorFH, (7,7), 10) # blurring the edges to make it smooth
    frameColorFH = cv2.addWeighted(frame,1,frameColorFH,0.4,0)

    frameLeftEye = createBox(frame, list5, 3, masked=True, cropped=False)
    frameColorLE = np.zeros_like(frameLeftEye)
    frameColorLE[:] = 226,226,255 # coloured frame 
    frameColorLE = cv2.bitwise_and(frameLeftEye, frameColorLE) # coloring the lips on the masked frame
    frameColorLE = cv2.GaussianBlur(frameColorLE, (7,7), 10) # blurring the edges to make it smooth
    frameColorLE = cv2.addWeighted(frameColorFH ,1,frameColorLE,0.4,0)

    frameLeftCheek = createBox(frame, list6, 3, masked=True, cropped=False)
    frameColorLC = np.zeros_like(frameLeftCheek)
    frameColorLC[:] = 88,46,255 # coloured frame 
    frameColorLC = cv2.bitwise_and(frameLeftCheek, frameColorLC) # coloring the lips on the masked frame
    frameColorLC = cv2.GaussianBlur(frameColorLC, (7,7), 10) # blurring the edges to make it smooth
    frameColorLC = cv2.addWeighted(frameColorLE ,1,frameColorLC,0.4,0)

    frameChin = createBox(frame, list7, 3, masked=True, cropped=False)
    frameColorChin = np.zeros_like(frameChin)
    frameColorChin[:] = 226,226,255 # coloured frame 
    frameColorChin = cv2.bitwise_and(frameChin, frameColorChin) # coloring the lips on the masked frame
    frameColorChin = cv2.GaussianBlur(frameColorChin, (7,7), 10) # blurring the edges to make it smooth
    frameColorChin = cv2.addWeighted(frameColorLC ,1,frameColorChin,0.4,0)

    frameRightCheek = createBox(frame, list8, 3, masked=True, cropped=False)
    frameColorRC = np.zeros_like(frameRightCheek)
    frameColorRC[:] = 88,46,255 # coloured frame 
    frameColorRC = cv2.bitwise_and(frameRightCheek, frameColorRC) # coloring the lips on the masked frame
    frameColorRC = cv2.GaussianBlur(frameColorRC, (7,7), 10) # blurring the edges to make it smooth
    frameColorRC = cv2.addWeighted(frameColorChin ,1,frameColorRC,0.4,0)

    frameRightEye = createBox(frame, list9, 3, masked=True, cropped=False)
    frameColorRE = np.zeros_like(frameRightEye)
    frameColorRE[:] = 226,226,255 # coloured frame 
    frameColorRE = cv2.bitwise_and(frameRightEye, frameColorRE) # coloring the lips on the masked frame
    frameColorRE = cv2.GaussianBlur(frameColorRE, (7,7), 10) # blurring the edges to make it smooth
    frameColorRE = cv2.addWeighted(frameColorRC ,1,frameColorRE,0.4,0)

    return frameColorRE


def rectangleShape(frame, list1):

    # rectangle forehead
    list10 = []
    list10.append(list1[103])
    list10.append(list1[68])
    list10.append(list1[63])
    list10.append(list1[104])
    list10.append(list1[69])
    list10.append(list1[108])
    list10.append(list1[151])
    list10.append(list1[337])
    list10.append(list1[299])
    list10.append(list1[333])
    list10.append(list1[293])
    list10.append(list1[298])
    list10.append(list1[332])
    list10.append(list1[297])
    list10.append(list1[338])
    list10.append(list1[10])
    list10.append(list1[109])
    list10.append(list1[67])
    list10 = np.array(list10)
        
    # LEFT UNDEREYE
    list11 = []
    list11.append(list1[31])
    list11.append(list1[233])
    list11.append(list1[205])
    list11 = np.array(list11)

    # RIGHT UNDEREYE
    list12 = []
    list12.append(list1[453])
    list12.append(list1[261])
    list12.append(list1[425])
    list12 = np.array(list12)

    # LEFT CHEEK
    list13 = []
    list13.append(list1[116])
    list13.append(list1[123])
    list13.append(list1[187])
    list13.append(list1[207])
    list13.append(list1[205])
    list13.append(list1[117])
    list13.append(list1[111])
    list13 = np.array(list13)

    # LEFT JAW
    list14 = []
    list14.append(list1[177])
    list14.append(list1[58])
    list14.append(list1[172])
    list14.append(list1[136])
    list14.append(list1[150])
    list14.append(list1[149])
    list14.append(list1[140])
    list14.append(list1[170])
    list14.append(list1[169])
    list14.append(list1[135])
    list14.append(list1[138])
    list14.append(list1[215])
    list14 = np.array(list14)

    # chin
    list15 = []
    list15.append(list1[208])
    list15.append(list1[171])
    list15.append(list1[175])
    list15.append(list1[396])
    list15.append(list1[428])
    list15.append(list1[421])
    list15.append(list1[200])
    list15.append(list1[201])
    list15 = np.array(list15)

    # RIGHT JAW
    list16 = []
    list16.append(list1[369])
    list16.append(list1[378])
    list16.append(list1[379])
    list16.append(list1[365])
    list16.append(list1[397])
    list16.append(list1[288])
    list16.append(list1[401])
    list16.append(list1[435])
    list16.append(list1[367])
    list16.append(list1[364])
    list16.append(list1[394])
    list16.append(list1[395])
    list16 = np.array(list16)

    # RIGHT CHEEK
    list17 = []
    list17.append(list1[345])
    list17.append(list1[340])
    list17.append(list1[346])
    list17.append(list1[425])
    list17.append(list1[427])
    list17.append(list1[411])
    list17.append(list1[352])
    list17 = np.array(list17)

    frameRecFH = createBox(frame, list10, 3, masked=True, cropped=False)
    frameColorRecFH = np.zeros_like(frameRecFH)
    frameColorRecFH[:] = 88,46,255 # coloured frame 
    frameColorRecFH = cv2.bitwise_and(frameRecFH, frameColorRecFH) # coloring the lips on the masked frame
    frameColorRecFH = cv2.GaussianBlur(frameColorRecFH, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecFH = cv2.addWeighted(frame ,1,frameColorRecFH,0.4,0)

    frameRecLE = createBox(frame, list11, 3, masked=True, cropped=False)
    frameColorRecLE = np.zeros_like(frameRecLE)
    frameColorRecLE[:] = 226,226,255 # coloured frame 
    frameColorRecLE = cv2.bitwise_and(frameRecLE, frameColorRecLE) # coloring the lips on the masked frame
    frameColorRecLE = cv2.GaussianBlur(frameColorRecLE, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecLE = cv2.addWeighted(frameColorRecFH ,1,frameColorRecLE,0.4,0)

    frameRecRE = createBox(frame, list12, 3, masked=True, cropped=False)
    frameColorRecRE = np.zeros_like(frameRecRE)
    frameColorRecRE[:] = 226,226,255 # coloured frame 
    frameColorRecRE = cv2.bitwise_and(frameRecRE, frameColorRecRE) # coloring the lips on the masked frame
    frameColorRecRE = cv2.GaussianBlur(frameColorRecRE, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecRE = cv2.addWeighted(frameColorRecLE ,1,frameColorRecRE,0.4,0)

    frameRecLC = createBox(frame, list13, 3, masked=True, cropped=False)
    frameColorRecLC = np.zeros_like(frameRecLC)
    frameColorRecLC[:] = 88,46,255 # coloured frame 
    frameColorRecLC = cv2.bitwise_and(frameRecLC, frameColorRecLC) # coloring the lips on the masked frame
    frameColorRecLC = cv2.GaussianBlur(frameColorRecLC, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecLC = cv2.addWeighted(frameColorRecRE ,1,frameColorRecLC,0.4,0)

    frameRecLJ = createBox(frame, list14, 3, masked=True, cropped=False)
    frameColorRecLJ = np.zeros_like(frameRecLJ)
    frameColorRecLJ[:] = 88,46,255 # coloured frame 
    frameColorRecLJ = cv2.bitwise_and(frameRecLJ, frameColorRecLJ) # coloring the lips on the masked frame
    frameColorRecLJ = cv2.GaussianBlur(frameColorRecLJ, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecLJ = cv2.addWeighted(frameColorRecLC ,1,frameColorRecLJ,0.4,0)

    frameRecChin = createBox(frame, list15, 3, masked=True, cropped=False)
    frameColorRecChin = np.zeros_like(frameRecChin)
    frameColorRecChin[:] = 226,226,255 # coloured frame 
    frameColorRecChin = cv2.bitwise_and(frameRecChin, frameColorRecChin) # coloring the lips on the masked frame
    frameColorRecChin = cv2.GaussianBlur(frameColorRecChin, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecChin = cv2.addWeighted(frameColorRecLJ ,1,frameColorRecChin,0.4,0)

    frameRecRJ = createBox(frame, list16, 3, masked=True, cropped=False)
    frameColorRecRJ = np.zeros_like(frameRecRJ)
    frameColorRecRJ[:] = 88,46,255 # coloured frame 
    frameColorRecRJ = cv2.bitwise_and(frameRecRJ, frameColorRecRJ) # coloring the lips on the masked frame
    frameColorRecRJ = cv2.GaussianBlur(frameColorRecRJ, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecRJ = cv2.addWeighted(frameColorRecChin ,1,frameColorRecRJ,0.4,0)

    frameRecRC = createBox(frame, list17, 3, masked=True, cropped=False)
    frameColorRecRC = np.zeros_like(frameRecRC)
    frameColorRecRC[:] = 88,46,255 # coloured frame 
    frameColorRecRC = cv2.bitwise_and(frameRecRC, frameColorRecRC) # coloring the lips on the masked frame
    frameColorRecRC = cv2.GaussianBlur(frameColorRecRC, (7,7), 10) # blurring the edges to make it smooth
    frameColorRecRC = cv2.addWeighted(frameColorRecRJ ,1,frameColorRecRC,0.4,0)

    return frameColorRecRC


# main argument

def applyContour(faceShape):

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
            points=[]
            list1 = list(range(468))

            for face_landmarks in results.multi_face_landmarks:
                # mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, mp_draw.DrawingSpec((0,255,0), 1, 1))

                for id,lm in enumerate(face_landmarks.landmark):
                    ih, iw, ic = frame.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    list1[id] = [x,y]
                    # cv2.putText(frame, str(id), (x,y), cv2.FONT_HERSHEY_TRIPLEX, 0.3, (0, 0, 255), 1)

            list1 = np.array(list1)

            if faceShape == "diamond_face":
                cv2.imshow("Contour", diamondShape(frame, list1))

            elif faceShape == "rectangle_face":
                cv2.imshow("Contour", rectangleShape(frame, list1))
            
            elif faceShape == "heart_face":
                cv2.imshow("Contour", draw.heartShape(frame, list1))
            
            else:
                cv2.imshow("Contour", draw.roundShape(frame, list1))

        # cv2.imshow("Frame", frame)
        key = cv2.waitKey(1)

        if key == 27:
            break

if __name__ == "__main__":
    applyContour("rectangle")
    
  
