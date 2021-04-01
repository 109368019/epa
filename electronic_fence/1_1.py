# coding=UTF-8

import cv2
import numpy as np
from matplotlib import pyplot as plt 

videoCapture = cv2.VideoCapture('2.mp4')

def opening(image,kernel):
    image = cv2.erode(image,kernel,1)    
    image = cv2.dilate(image,kernel,1)
    return image

frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0 
success, frame = videoCapture.read()    #讀幀
initFrame = frame                       #紀錄第一張Frame (RGB)
seq = np.expand_dims(cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)[:,:,2], axis = 2)   #建立seq(YUV)
ff=3050


hsv =np.zeros((1000,480, 720,3), np.float32)
z=0
while success :
    print(count)

    count = count+1

    if count > frame_count-1:
        break
    before = cv2.cvtColor(frame,cv2.COLOR_RGB2HSV)      #紀錄前一幀
    success, frame = videoCapture.read()#讀取下一幀
    #cv2.imshow('windows', frame)        #show 原始圖

    if  count >ff:
        # frameBlur = frame
        frameBlur = cv2.medianBlur(frame,3) #中值濾波 rgb 
        #cv2.imshow('frameBlur', frameBlur)
        frameHSV = cv2.cvtColor(frameBlur,cv2.COLOR_RGB2HSV)/255   #轉yuv並轉0~1
        if z>=1000:
            z=0
        if count<ff+1000:
            hsv[z]=frameHSV
            z=z+1
        if count==ff+1000:
            p=sum(hsv[0:1000,:,:,2])/1000

        if count>ff+1000:
            frameDiff=abs(p[:,:]-frameHSV[:,:,2])
            frameDiff=frameDiff/frameDiff.max()
            ret,frameBinary = cv2.threshold(frameDiff,0.5,1,cv2.THRESH_BINARY) 

            img=(frameBinary*255).astype(np.uint8)
            cv2.imshow('n',img)
            img = cv2.erode(img, None, iterations=1)    #侵蝕膨脹去雜訊
            img = cv2.dilate(img, None, iterations=10)
            img = cv2.erode(img, None, iterations=5)
            
			
            _,ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for ctr in ctrs:
                x, y, w, h = cv2.boundingRect(ctr)
                boxes.append([x, y, w, h])     #每個輪廓的範圍存進方形的bounding box

            for i, box in enumerate(boxes):
                top_left     = (box[0], box[1])
                bottom_right = (box[0] + box[2], box[1] + box[3])
                cv2.rectangle(frameBlur, top_left, bottom_right, (0,255,0), 2)     #畫bounding box在彩色畫面上
                ROI_img = frameBlur[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]     #把bounding box的ROI範圍截下存成新圖(阿杜說可當訓練樣本)
                cv2.imwrite('data1_1/data_%d_%d.png' %(count, i), ROI_img)     #count是當前第幾幀， i是當前幀數的第幾個框

            cv2.imshow('dilate',img)
            cv2.imshow('frameBlur', frameBlur)
            cv2.waitKey(100)

            p=p-hsv[z,:,:,2]/1000+frameHSV[:,:,2]/1000
            hsv[z,:,:,2]=frameHSV[:,:,2]
            z=z+1


    


