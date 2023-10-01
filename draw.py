import cv2 
import  numpy as np
import mediapipe as mp

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



def roundShape(frame, list1):
    #ROUND FACE

    pts2 = [list1[32],list1[194], list1[83], list1[18], list1[313], list1[418], list1[262],list1[428],list1[199], list1[208]]
    pts3 = [list1[10], list1[337], list1[9], list1[108]]
    pts4 = [list1[205], list1[111], list1[121]]
    pts5 = [list1[425],list1[340],list1[350]]
    pts6 = [list1[227], list1[137], list1[147], list1[213], list1[192], list1[214], list1[210], list1[212], list1[207], 
            list1[187], list1[123], list1[116], list1[143], list1[156], list1[70], list1[63], list1[105], list1[66], 
            list1[69], list1[109], list1[67], list1[103], list1[54], list1[71], list1[139], list1[34]]
    pts7 = [list1[338],list1[299],list1[296],list1[334],list1[293],list1[300],list1[383],list1[372],list1[345],
            list1[352],list1[411],list1[427],list1[432],list1[430],list1[434],list1[416],list1[433],list1[376],
            list1[366],list1[447],list1[264],list1[368],list1[301],list1[284],list1[332],list1[297]]

    pts2 = np.array(pts2)
    pts3 = np.array(pts3)
    pts4 = np.array(pts4)
    pts5 = np.array(pts5)
    pts6 = np.array(pts6)
    pts7 = np.array(pts7)

    frameRoundChin = createBox(frame, pts2, 3, masked=True,cropped=False)
    frameRFH = createBox(frame, pts3, 3, masked=True,cropped=False)
    frameRCheekL = createBox(frame, pts4, 3, masked=True,cropped=False)
    frameRCheekR = createBox(frame, pts5, 3, masked=True,cropped=False)
    frameRContourL = createBox(frame, pts6, 3, masked=True,cropped=False)
    frameRContourR = createBox(frame, pts7, 3, masked=True,cropped=False)

    frameColorRoundChin = np.zeros_like(frameRoundChin)
    frameColorRoundChin[:] = 226,226,225 # coloured frame 
    frameColorRoundChin = cv2.bitwise_and(frameRoundChin, frameColorRoundChin) # coloring the lips on the masked frame
    frameColorRoundChin = cv2.GaussianBlur(frameColorRoundChin, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frame,1,frameColorRoundChin,0.4,0)

    frameColorRFH = np.zeros_like(frameRFH)
    frameColorRFH[:] = 226,226,225 # coloured frame 
    frameColorRFH = cv2.bitwise_and(frameRFH, frameColorRFH) # coloring the lips on the masked frame
    frameColorRFH = cv2.GaussianBlur(frameColorRFH, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorRFH,0.4,0)

    frameColorRCheekL = np.zeros_like(frameRCheekL)
    frameColorRCheekL[:] = 226,226,225 # coloured frame 
    frameColorRCheekL = cv2.bitwise_and(frameRCheekL, frameColorRCheekL) # coloring the lips on the masked frame
    frameColorRCheekL = cv2.GaussianBlur(frameColorRCheekL, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorRCheekL,0.4,0)

    frameColorRCheekR = np.zeros_like(frameRCheekR)
    frameColorRCheekR[:] = 226,226,225 # coloured frame 
    frameColorRCheekR = cv2.bitwise_and(frameRCheekR, frameColorRCheekR) # coloring the lips on the masked frame
    frameColorRCheekR = cv2.GaussianBlur(frameColorRCheekR, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorRCheekR,0.4,0)

    frameColorRContourL = np.zeros_like(frameRContourL)
    frameColorRContourL[:] = 88,46,255 # coloured frame 
    frameColorRContourL = cv2.bitwise_and(frameRContourL, frameColorRContourL) # coloring the lips on the masked frame
    frameColorRContourL = cv2.GaussianBlur(frameColorRContourL, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorRContourL,0.4,0)

    frameColorRContourR = np.zeros_like(frameRContourR)
    frameColorRContourR[:] = 88,46,255# coloured frame 
    frameColorRContourR = cv2.bitwise_and(frameRContourR, frameColorRContourR) # coloring the lips on the masked frame
    frameColorRContourR = cv2.GaussianBlur(frameColorRContourR, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorRContourR,0.4,0)

    return frameColor
    #END ROUND FACE


def heartShape(frame, list1):
    #START HEART FACE

    Hpts1 = np.array([list1[108], list1[10], list1[337], list1[8]])
    Hpts2 = np.array([list1[194], list1[83], list1[18], list1[313], list1[418], list1[421], list1[200], list1[201]])
    Hpts3 = np.array([list1[32], list1[208], list1[199], list1[428], list1[262], list1[369], list1[396], list1[175], 
            list1[171], list1[140]])
    Hpts4 = np.array([list1[205], list1[232], list1[31], list1[156], list1[111]])
    Hpts5 = np.array([list1[425], list1[452], list1[261], list1[383], list1[340]])
    Hpts6 = np.array([list1[54], list1[68], list1[71], list1[139], list1[34], list1[116], 
            list1[207], list1[192], list1[213], list1[137], list1[234], list1[127], list1[162], list1[21]])
    Hpts7 = np.array([list1[284], list1[251], list1[389], list1[356], list1[454], list1[366], list1[433], 
            list1[416], list1[427], list1[345], list1[264], list1[368], list1[301], list1[298]])

    frameHFH = createBox(frame, Hpts1, 3, masked=True,cropped=False)
    frameHChin1  = createBox(frame, Hpts2, 3, masked=True,cropped=False)
    frameHChin2  = createBox(frame, Hpts3, 3, masked=True,cropped=False)
    frameHCheekL = createBox(frame, Hpts4, 3, masked=True,cropped=False)
    frameHCheekR = createBox(frame, Hpts5, 3, masked=True,cropped=False)
    frameHContourL = createBox(frame, Hpts6, 3, masked=True,cropped=False)
    frameHContourR = createBox(frame, Hpts7, 3, masked=True,cropped=False)

    frameColorHFH = np.zeros_like(frameHFH)
    frameColorHFH[:] = 226,226,225 # coloured frame 
    frameColorHFH = cv2.bitwise_and(frameHFH, frameColorHFH) # coloring the lips on the masked frame
    frameColorHFH = cv2.GaussianBlur(frameColorHFH, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frame,1,frameColorHFH,0.4,0)

    frameColorHChin1 = np.zeros_like(frameHChin1)
    frameColorHChin1[:] = 226,226,225 # coloured frame 
    frameColorHChin1 = cv2.bitwise_and(frameHChin1, frameColorHChin1) # coloring the lips on the masked frame
    frameColorHChin1 = cv2.GaussianBlur(frameColorHChin1, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorHChin1,0.4,0)

    frameColorHChin2 = np.zeros_like(frameHChin2)
    frameColorHChin2[:] = 88,46,255 # coloured frame 
    frameColorHChin2 = cv2.bitwise_and(frameHChin2, frameColorHChin2) # coloring the lips on the masked frame
    frameColorHChin2 = cv2.GaussianBlur(frameColorHChin2, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorHChin2,0.4,0)

    frameColorHCheekL = np.zeros_like(frameHCheekL)
    frameColorHCheekL[:] = 226,226,225 # coloured frame 
    frameColorHCheekL = cv2.bitwise_and(frameHCheekL, frameColorHCheekL) # coloring the lips on the masked frame
    frameColorHCheekL = cv2.GaussianBlur(frameColorHCheekL, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorHCheekL,0.4,0)

    frameColorHCheekR = np.zeros_like(frameHCheekR)
    frameColorHCheekR[:] = 226,226,225 # coloured frame 
    frameColorHCheekR = cv2.bitwise_and(frameHCheekR, frameColorHCheekR) # coloring the lips on the masked frame
    frameColorHCheekR = cv2.GaussianBlur(frameColorHCheekR, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorHCheekR,0.4,0)

    frameColorHContourL = np.zeros_like(frameHContourL)
    frameColorHContourL[:] = 88,46,255 # coloured frame 
    frameColorHContourL = cv2.bitwise_and(frameHContourL, frameColorHContourL) # coloring the lips on the masked frame
    frameColorHContourL = cv2.GaussianBlur(frameColorHContourL, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorHContourL,0.4,0)

    frameColorHContourR = np.zeros_like(frameHContourR)
    frameColorHContourR[:] = 88,46,255 # coloured frame 
    frameColorHContourR = cv2.bitwise_and(frameHContourR, frameColorHContourR) # coloring the lips on the masked frame
    frameColorHContourR = cv2.GaussianBlur(frameColorHContourR, (7,7), 10) # blurring the edges to make it smooth
    frameColor = cv2.addWeighted(frameColor,1,frameColorHContourR,0.4,0)

    return frameColor

