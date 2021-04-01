# import os
#
# import cv2
# import numpy as np
# import numba as nb
# from para.parameter import Parameter
#
# para = Parameter()
#
#
# def opening(image, kernel):
#     image = cv2.erode(image, kernel, 1)
#     image = cv2.dilate(image, kernel, 1)
#     return image
#
# @nb.jit
# def counting(z, hsv, v_count):
#     # 統計不同亮度值的數量
#     for x in range(frame_x):
#         for y in range(frame_y):
#             v_count[x, y, int(hsv[z, x, y, 2] * 255)] = v_count[x, y, int(hsv[z, x, y, 2] * 255)] + 1
#     return v_count
#
#
# @nb.jit
# def up_counting(z, hsv, v_count, frameHSV):
#     # 更新統計不同亮度值的數量
#     for x in range(frame_x):
#         for y in range(frame_y):
#             v_count[x, y, int(hsv[z, x, y, 2] * 255)] = v_count[x, y, int(hsv[z, x, y, 2] * 255)] - 1
#             v_count[x, y, int(frameHSV[x, y, 2] * 255)] = v_count[x, y, int(frameHSV[x, y, 2] * 255)] + 1
#     return v_count
#
#
# video_list = [i for i in os.listdir("../video/") if (i[-3::] == "mp4")]
# for num, video_name in enumerate(video_list):
#     print("{}. {}".format(num, video_name))
# input_video = "../video/" + video_list[int(input('Select video: '))]
# video_capture = cv2.VideoCapture(input_video)
#
# frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
# count = 0
# statistics_index = 0
#
# success, frame = video_capture.read()  # 讀幀
# frame_x = frame.shape[0]        #480
# frame_y = frame.shape[1]        #720
#
# v_count = np.zeros((frame_x, frame_y, 257), np.uint16)          #統計直方圖
# print_img = np.zeros((frame_x, frame_y), np.uint8)
# save_frameDiff2 = np.zeros((frame_x, frame_y), np.uint8)
# hsv = np.zeros((para.arg_count, frame_x, frame_y, 3), np.float32)
# while success:
#     print(count)
#     count = count + 1
#     if count > frame_count - 1:
#         break
#     success, frame = video_capture.read()  # 讀取下一幀
#     if count > para.start_frame:
#         frameBlur = cv2.medianBlur(frame, 3)  # 中值濾波 rgb
#         # frameBlur = cv2.GaussianBlur(frame, 3)  # 高斯濾波 rgb
#         frameHSV = cv2.cvtColor(frameBlur, cv2.COLOR_RGB2HSV)
#         if statistics_index >= para.arg_count:
#             statistics_index = 0
#         if count < para.start_frame + para.arg_count:
#             hsv[statistics_index] = frameHSV
#             v_count = counting(statistics_index, hsv, v_count)  # 統計初始值方圖
#             statistics_index = statistics_index + 1
#         if count == para.start_frame + para.arg_count:
#             p = sum(hsv[0:para.arg_count, :, :, 2]) / para.arg_count    #avg value
#             v_count = counting(statistics_index, hsv, v_count)
#         if count > para.start_frame + para.arg_count:
#             frameDiff = abs(p[:, :] - frameHSV[:, :, 2])
#             frameDiff = frameDiff / frameDiff.max()
#             frameDiff2 = np.zeros((frame_x, frame_y), np.uint8)
#             ret, frameBinary = cv2.threshold(frameDiff, 0.4, 1, cv2.THRESH_BINARY)  #?? threshold
#             k = 5  #
# ########################################################繪製二值化圖
#             for x in range(frame_x):
#                 for y in range(frame_y):
#                     if v_count[x, y, int(frameHSV[x, y, 2]) + 1] < k and v_count[
#                         x, y, int(frameHSV[x, y, 2]) - 1] < k and v_count[x, y, int(frameHSV[x, y, 2])] < k:
#                         frameDiff2[x, y] = 255
#             cv2.imshow('count', frameDiff2)
#             # img = (frameBinary * 255).astype(np.uint8)
#  ########################################################做AND
#             img = np.zeros((frame_x, frame_y), np.uint8)
#             img = frameDiff2
#             print_img = save_frameDiff2 & frameDiff2
#
#             # for i in range(frame_x):
#             #     for j in range(frame_y):
#             #         if save_frameDiff2[i, j] == frameDiff2[i, j]:
#             #             print_img[i, j] = frameDiff2[i, j]
#             save_frameDiff2 = frameDiff2
#             # img = cv2.dilate(img, None, iterations=1)
#             img = print_img
#             cv2.imshow('and', img)
#             cv2.waitKey(10)
#
#
#
#
#
#
#
#
#
# coding=UTF-8
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numba as nb
# from matplotlib import pyplot as plt
import time

# videoCapture = cv2.VideoCapture('../video/2.mp4')
# 0814
video_list = [i for i in os.listdir("../video/") if (i[-3::] == "mp4")]
for num, video_name in enumerate(video_list):
    print("{}. {}".format(num, video_name))
input_video = "../video/" + video_list[int(input('Select video: '))]
videoCapture = cv2.VideoCapture(input_video)
# coding=UTF-8

import cv2
import numpy as np
import numba as nb
# from matplotlib import pyplot as plt
import time


# videoCapture = cv2.VideoCapture('2.mp4')


def opening(image, kernel):
    image = cv2.erode(image, kernel, 1)
    image = cv2.dilate(image, kernel, 1)
    return image


@nb.jit
def counting(z, hsv, v_count):
    # 統計不同亮度值的數量
    for x in range(frame_x):
        for y in range(frame_y):
            v_count[x, y, int(hsv[z, x, y, channel] * 255)] = v_count[x, y, int(hsv[z, x, y, channel] * 255)] + 1
    return v_count


@nb.jit
def up_counting(z, hsv, v_count, frameHSV):
    # 更新統計不同亮度值的數量
    for x in range(frame_x):
        for y in range(frame_y):
            v_count[x, y, int(hsv[z, x, y, channel] * 255)] = v_count[x, y, int(hsv[z, x, y, channel] * 255)] - 1
            v_count[x, y, int(frameHSV[x, y, channel] * 255)] = v_count[x, y, int(frameHSV[x, y, channel] * 255)] + 1
    return v_count


def check_mean(img, channel, x1, x2, y1, y2):
    block = img[x1:x2, y1:y2, channel]
    mean = block.mean()
    cv2.imshow("block", block)
    print("{} mean: {}".format(channel, mean))


frame_count = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
count = 0
success, frame = videoCapture.read()  # 讀幀
frame_x = frame.shape[0]  # 480
frame_y = frame.shape[1]  # 720
initFrame = frame  # 紀錄第一張Frame (RGB)
seq = np.expand_dims(cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:, :, 2], axis=2)  # 建立seq(YUV)
start_frame = 5050
arg_count = 1000  # 總統計數量
channel = 2
v_count = np.zeros((frame_x, frame_y, 257), np.uint16)  # 統計直方圖
print_img = np.zeros((frame_x, frame_y), np.uint8)
save_frameDiff2 = np.zeros((frame_x, frame_y), np.uint8)
hsv = np.zeros((arg_count, frame_x, frame_y, 3), np.float32)
statistics_index = 0
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('./output_video/outputVideo.avi', fourcc, 6.99, (frame.shape[1], frame.shape[0]))

while success:
    print(count)
    count = count + 1

    if count > frame_count - 1:
        break
    before = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)  # 紀錄前一幀
    success, frame0 = videoCapture.read()  # 讀取下一幀
    # cv2.imshow('windows', frame)        #show 原始圖

    # frame1 = frame0[0:240, 0:360, :]
    # frame2 = frame0[0:240, 360:720, :]
    # frame3 = frame0[240:480, 0:360, :]
    # frame4 = frame0[240:480, 360:720, :]

    frame = frame0
    if count > start_frame:
        # frameBlur = frame
        frameBlur = cv2.medianBlur(frame, 3)  # 中值濾波 rgb
        # cv2.imshow('frameBlur', frameBlur)
        frameHSV = cv2.cvtColor(frameBlur, cv2.COLOR_RGB2HSV) / 255
        if statistics_index >= arg_count:
            statistics_index = 0
        if count < start_frame + arg_count:
            hsv[statistics_index] = frameHSV
            v_count = counting(statistics_index, hsv, v_count)  # 統計初始值方圖
            '''
            for x in range(240):
                for y in range(360):
                    v_count[x,y,int(hsv[z,x,y,2]*255)]=v_count[x,y,int(hsv[z,x,y,2]*255)]+1
                    if x==0 and y==0:
                        print ('%d %d' %(int(hsv[z,x,y,2]*255),z))
            '''
            statistics_index = statistics_index + 1
        if count == start_frame + arg_count:
            # p = sum(hsv[0:arg_count, :, :, channel]) / arg_count
            v_count = counting(statistics_index, hsv, v_count)
        if count > start_frame + arg_count:
            # frameDiff = abs(p[:, :] - frameHSV[:, :, channel])
            # frameDiff = frameDiff / frameDiff.max()
            frameDiff2 = np.zeros((frame_x, frame_y, 1), np.uint8)
            # ret, frameBinary = cv2.threshold(frameDiff, 0.4, 1, cv2.THRESH_BINARY)
            k = 5  #

            ########################################################繪製二值化圖
            # for x in range(480):
            #     for y in range(720):
            #         if v_count[x, y, int(frameHSV[x, y, channel] * 255)] < k:
            #             frameDiff2[x, y, 0] = 255
            xxx = frameDiff2[: ,:, 0]
            yyy = np.take_along_axis(v_count, (frameHSV[:, :, channel] * 255).reshape(480, 720, 1).astype(int), axis=2) < k
            xxx[yyy.reshape(480, 720)] = 255
            frameDiff2[:, :, 0] = xxx
                    # v_count[x,y,int(frameHSV[x,y,2]*255)+1] <k and v_count[x,y,int(frameHSV[x,y,2]*255)-1]<k and  v_count[x,y,int(frameHSV[x,y,2]*255)]<k
            cv2.imshow('count', frameDiff2)
            # img = (frameBinary * 255).astype(np.uint8)
            # img= cv2.medianBlur(frameDiff2,3)#中值
            ########################################################做AND
            img = frameDiff2
            for i in range(frame_x):
                for j in range(frame_y):
                    if save_frameDiff2[i, j] == frameDiff2[i, j]:
                        print_img[i, j] = frameDiff2[i, j]
            # print_img[save_frameDiff2==frameDiff2] = frameDiff2[save_frameDiff2==frameDiff2]
            save_frameDiff2 = frameDiff2
            # img = cv2.dilate(img, None, iterations=1)
            img = print_img
            cv2.imshow('and', img)

            ########################################################侵蝕膨脹
            img = cv2.dilate(img, None, iterations=1)  # 侵蝕膨脹去雜訊
            img = cv2.erode(img, None, iterations=4)  # 侵蝕膨脹去雜訊
            img = cv2.dilate(img, None, iterations=11)
            # img = cv2.erode(img, None, iterations=5)

            ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, ctrs, -1, (0, 0, 255), 2)
            cv2.imshow('frame', frame)

            boxes = []
            for ctr in ctrs:
                x, y, w, h = cv2.boundingRect(ctr)
                boxes.append([x, y, w, h])  # 每個輪廓的範圍存進方形的bounding box

            for i, box in enumerate(boxes):
                top_left = (box[0], box[1])
                bottom_right = (box[0] + box[2], box[1] + box[3])
                cv2.rectangle(frameBlur, top_left, bottom_right, (0, 255, 0), 2)  # 畫bounding box在彩色畫面上
                ROI_img = frame[top_left[1]:bottom_right[1],
                          top_left[0]:bottom_right[0]]  # 把bounding box的ROI範圍截下存成新圖(阿杜說可當訓練樣本)
                # cv2.imwrite('data1_2/data_%d_%d.png' %(count , i), ROI_img)     #count是當前第幾幀， i是當前幀數的第幾個框
            check_mean(frameHSV, channel, 121, 209, 365, 421)
            out1.write(frameBlur)

            # plt.figure(figsize=(15, 10), dpi=100, linewidth=2)
            # plt.plot(v_count[121, 365, :], 's-', color='r', label="TSMC")
            # plt.show()

            cv2.imshow('dilate', img)
            cv2.imshow('frameBlur', frameBlur)
            cv2.waitKey(10)
            # print(sum(v_count[80,120,:]))
            # p = p - hsv[statistics_index, :, :, channel] / arg_count + frameHSV[:, :, channel] / arg_count
            v_count = up_counting(statistics_index, hsv, v_count, frameHSV)  # 更新值方圖
            hsv[statistics_index, :, :, channel] = frameHSV[:, :, channel]
            statistics_index = statistics_index + 1
