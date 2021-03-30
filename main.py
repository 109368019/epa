import cv2 as cv2
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN
import os
from para.parameter import Parameter

def opening(image, kernel1, kernel2):
    image = cv2.erode(image, kernel1, 1)
    image = cv2.dilate(image, kernel2, 1)
    # image = cv2.erode(image, kernel2, 1)
    # image = cv2.dilate(image, kernel1, 1threshold_value)
    return image


def main():
    # print('Finish')

    # parameter=================================================
    # para.count = 0
    # para.outset_frame = 0
    # para.cycle_point = 0
    # para.channel = 2  # H:0 S:1 V:2
    # para.window_length = 10
    #
    # para.kernel_size1 = (5, 5)
    # para.kernel_size2 = (7, 7)cv2.destroyAllWindows()
    # para.threshold_value = 0.07
    # para.median_blur_value = 3
    # para.wait_time = 100
    # from para import para_4723_0306_150000 as para
    # para.count, para.outset_frame, para.cycle_point, para.channel, para.window_length, para.kernel_size1, para.kernel_size2, para.threshold_value, para.median_blur_value, para.wait_time, input_video = para.import_para()

    para = Parameter()

    video_list = [i for i in os.listdir("./") if (i[-3::] == "mp4")]
    for num, video_name in enumerate(video_list):
        print("{}. {}".format(num, video_name))
    input_video = "./" + video_list[int(input('Select video: '))]
    # print(video_list[3])

    # input_video = './綠環境-(A15)家電冷氣機設備投入口及處理線、冷媒吸取區-20200331114524-20200306-130000-movie.mp4'
    # input_video = './綠環境-(A15)家電冷氣機設備投入口及處理線、冷媒吸取區-20200331114640-20200306-150000-movie.mp4'
    # input_video = './綠環境-(A15)家電冷氣機設備投入口及處理線、冷媒吸取區-20200331114723-20200306-150000-movie.mp4'
    # input_video = '瑞原環保-09冷媒回收設備區、10廢物品拆解線、13冰箱拆解線及冷媒吸取區、14冷媒回收設備區衍生物貯存區-20200717161702-20200623-080000-movie.mp4'
    # input_video = './綠環境-(B5)家電冷氣機冷媒吸取區、冷氣機及洗衣機處理線-20200331115136-20200306-110000-movie.mp4'
    # input_video = './20200416大東方-鏡頭13到鏡頭16-20200416-140000.mp4_20200423_091228.mkv'
    # input_video = './20200416大東方-鏡頭13到鏡頭16-20200416-100000.mp4_20200423_090536.mkv'
    # input_video = './5071922111_20200902_090000.mp4'
    # input_video = '/home/spie/Desktop/EPA/Test/0306/綠電楊梅-01廠區大門、02廢物品已_待認證區A、G、S、03廢物品已_待認證區H、G、S、T、04廢物品已_待認證區A、B、C廢物品已_待認證區D、F-20210312-150000.mp4'

    video_capture = cv2.VideoCapture(input_video)

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = video_capture.read()  # 讀幀

    hsv_window = np.zeros((para.window_length, 480, 720, 3), np.float32)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    out1 = cv2.VideoWriter('outputVideo.avi', fourcc, 6.99, (frame.shape[1], frame.shape[0]))
    while success:
        print(para.count)

        para.count += 1

        if para.count > frame_count - 1:
            break
        success, frame = video_capture.read()  # 讀取下一幀

        if para.count > para.outset_frame:
            frame_blur = cv2.medianBlur(frame, para.median_blur_value)  # 中值濾波 rgb
            current_frame_hsv = cv2.cvtColor(
                frame_blur, cv2.COLOR_RGB2HSV) / 255  # 轉HSV並轉0~1
            if para.cycle_point >= para.window_length:
                para.cycle_point = 0
            if para.count < para.outset_frame + para.window_length:
                hsv_window[para.cycle_point] = current_frame_hsv
                para.cycle_point += 1
            if para.count == para.outset_frame + para.window_length:
                avg_value = sum(hsv_window[0:para.window_length, :, :, para.channel]) / para.window_length

            if para.count > para.outset_frame + para.window_length:
                frame_diff = abs(avg_value[:, :] - current_frame_hsv[:, :, para.channel])
                frame_diff = frame_diff / frame_diff.max()

                ret, frame_binary = cv2.threshold(
                    frame_diff, para.threshold_value, 1, cv2.THRESH_BINARY)

                img = (frame_binary * 255).astype(np.uint8)

                kernel1 = np.ones(para.kernel_size1, np.uint8)  # kernel
                kernel2 = np.ones(para.kernel_size2, np.uint8)  # kernel

                img = opening(img, kernel1, kernel2)

                clusteringDH(current_frame_hsv, img,out1,frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(para.wait_time) == ord('w'):
                    cv2.waitKey(0)

                avg_value = avg_value - hsv_window[para.cycle_point, :, :, para.channel] / \
                            para.window_length + current_frame_hsv[:, :, para.channel] / para.window_length

                # statisticsBackground(hsv_window, avg_value, current_frame_hsv, para.channel, kernel1, kernel2)
                hsv_window[para.cycle_point, :, :, para.channel] = current_frame_hsv[:, :, para.channel]
                para.cycle_point += 1


def clusteringDH(current_frame_HSV, img, out1, frame, sat_th=0.17):
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    vimg = (current_frame_HSV[:, :, 2] * 255).astype(np.uint8)
    v2_grayimg = cv2.merge([vimg, vimg, vimg])
    result = np.array(img).astype(np.uint8)
    
    result1 = v2_grayimg
    contours, hierarchy = cv2.findContours(result.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v2_grayimg, contours, -1, (0, 0, 255), 2)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    x, y = np.where(result == 255)
    data = np.dstack([x, y])
    data = data[0]
    
    clustering = DBSCAN(eps=5, min_samples=70).fit(data)
    
    last = np.setdiff1d(np.unique(clustering.labels_), np.array([-1]))
    
    for i in last:
        a = np.where(clustering.labels_ == i)
        class_map = np.zeros((current_frame_HSV.shape[0], current_frame_HSV.shape[1]))
        xy = [(x[i], y[i]) for i in a]
        for i, j in xy:
            class_map[i, j] = 1

        S = current_frame_HSV[:, :, 1]
        H = current_frame_HSV[:, :, 0]
        V = current_frame_HSV[:, :, 2]
        if(S[class_map == 1].mean() < sat_th):
            result1[class_map == 1] = 0
            cv2.imshow('Test', result1)
            cv2.imshow('obj', current_frame_HSV[min(x[a]):max(x[a]), min(y[a]):max(y[a]), 1])
            print('mean saturation value: ', H[class_map == 1].mean(), S[class_map == 1].mean(), V[class_map == 1].mean())
            cv2.waitKey(0)
            cv2.destroyWindow("obj")

    out1.write(frame)
    cv2.imshow('Test', result1)


def statisticsBackground(hsvWindow, avgValue, currentFrameHSV, para.channel, kernel1, kernel2):
    std_value = np.std(hsvWindow[:, :, :, para.channel], axis=0)
    std_upper = avgValue + 2 * std_value + 0.01
    std_lower = avgValue - 2 * std_value - 0.01
    moving_obj_filter = (currentFrameHSV[:, :, para.channel] <= std_upper) & (currentFrameHSV[:, :, para.channel] >= std_lower)
    # print(moving_obj_filter)
    # print(frameHSV[:, :, para.channel])
    # print(std_upper)
    # print(std_lower)
    now = currentFrameHSV[:, :, para.channel].copy()
    now[moving_obj_filter] = 0
    now[~moving_obj_filter] = 1
    now = opening(now, kernel1, kernel2)
    cv2.imshow('statisticsBackground', now)


main()
cv2.destroyAllWindows()
