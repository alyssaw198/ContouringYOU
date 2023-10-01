# for tuple in list(mp_face_mesh.FACEMESH_LIPS):
            # points.append(list(tuple))

        # print(points)

        # print(list1)

        list2 = [] # left eyebrow

        list2.append(list1[70])
        list2.append(list1[63])
        list2.append(list1[105])
        list2.append(list1[66])
        list2.append(list1[107])
        list2.append(list1[55])
        list2.append(list1[65])
        list2.append(list1[52])
        list2.append(list1[53])
        list2.append(list1[46])


        list3 = [] # left eyelid
        
        list3.append(list1[33])
        list3.append(list1[130])
        list3.append(list1[247])
        list3.append(list1[30])
        list3.append(list1[29])
        list3.append(list1[27])
        list3.append(list1[28])
        list3.append(list1[56])
        list3.append(list1[190])
        list3.append(list1[133])
        list3.append(list1[173])
        list3.append(list1[157])
        list3.append(list1[158])
        list3.append(list1[159])
        list3.append(list1[160])
        list3.append(list1[161])
        list3.append(list1[246])

        list3 = np.array(list3)

        # if (answer == "diamond"):



        # diamond forehead
        list4 = []
        list4.append(list1[108])
        list4.append(list1[9])
        list4.append(list1[337])
        list4.append(list1[10])
        list4 = np.array(list4)

        # diamond left undereye
        list5 = []
        list5.append(list1[31])
        list5.append(list1[233])
        list5.append(list1[205])
        list5 = np.array(list5)

        # diamond right undereye
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

        list7 = []
        list7.append(list1[208])
        list7.append(list1[175])
        list7.append(list1[428])
        list7.append(list1[200])
        list7 = np.array(list7)

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

        list9 = []
        list9.append(list1[453])
        list9.append(list1[261])
        list9.append(list1[425])
        list9 = np.array(list9)
        
        
     
       


        


        # list2 = np.array(list2)

        # points = np.array(points) 
        list1 = np.array(list1)
        
        
        # list2.append(list1[70])
        

        frameForeHead = createBox(frame, list4, 3, masked=True, cropped=False)
        frameColorFH = np.zeros_like(frameForeHead)
        frameColorFH[:] = 162,134,253 # coloured frame 
        frameColorFH = cv2.bitwise_and(frameForeHead, frameColorFH) # coloring the lips on the masked frame
        frameColorFH = cv2.GaussianBlur(frameColorFH, (7,7), 10) # blurring the edges to make it smooth
        frameColorFH = cv2.addWeighted(frame,1,frameColorFH,0.4,0)

        frameLeftEye = createBox(frame, list5, 3, masked=True, cropped=False)
        frameColorLE = np.zeros_like(frameLeftEye)
        frameColorLE[:] = 162,134,253 # coloured frame 
        frameColorLE = cv2.bitwise_and(frameLeftEye, frameColorLE) # coloring the lips on the masked frame
        frameColorLE = cv2.GaussianBlur(frameColorLE, (7,7), 10) # blurring the edges to make it smooth
        frameColorLE = cv2.addWeighted(frameColorFH ,1,frameColorLE,0.4,0)

        frameLeftCheek = createBox(frame, list6, 3, masked=True, cropped=False)
        frameColorLC = np.zeros_like(frameLeftCheek)
        frameColorLC[:] = 162,134,253 # coloured frame 
        frameColorLC = cv2.bitwise_and(frameLeftCheek, frameColorLC) # coloring the lips on the masked frame
        frameColorLC = cv2.GaussianBlur(frameColorLC, (7,7), 10) # blurring the edges to make it smooth
        frameColorLC = cv2.addWeighted(frameColorLE ,1,frameColorLC,0.4,0)

        frameChin = createBox(frame, list7, 3, masked=True, cropped=False)
        frameColorChin = np.zeros_like(frameChin)
        frameColorChin[:] = 162,134,253 # coloured frame 
        frameColorChin = cv2.bitwise_and(frameChin, frameColorChin) # coloring the lips on the masked frame
        frameColorChin = cv2.GaussianBlur(frameColorChin, (7,7), 10) # blurring the edges to make it smooth
        frameColorChin = cv2.addWeighted(frameColorLC ,1,frameColorChin,0.4,0)

        frameRightCheek = createBox(frame, list8, 3, masked=True, cropped=False)
        frameColorRC = np.zeros_like(frameRightCheek)
        frameColorRC[:] = 162,134,253 # coloured frame 
        frameColorRC = cv2.bitwise_and(frameRightCheek, frameColorRC) # coloring the lips on the masked frame
        frameColorRC = cv2.GaussianBlur(frameColorRC, (7,7), 10) # blurring the edges to make it smooth
        frameColorRC = cv2.addWeighted(frameColorChin ,1,frameColorRC,0.4,0)

        frameRightEye = createBox(frame, list9, 3, masked=True, cropped=False)
        frameColorRE = np.zeros_like(frameRightEye)
        frameColorRE[:] = 162,134,253 # coloured frame 
        frameColorRE = cv2.bitwise_and(frameRightEye, frameColorRE) # coloring the lips on the masked frame
        frameColorRE = cv2.GaussianBlur(frameColorRE, (7,7), 10) # blurring the edges to make it smooth
        frameColorRE = cv2.addWeighted(frameColorRC ,1,frameColorRE,0.4,0)


        # cv2.imshow('Lips', frameLips)
        # cv2.imshow("Colored", frameColorLips)
        cv2.imshow("ColoredLE", frameColorRE)