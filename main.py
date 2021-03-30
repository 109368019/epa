import cv2 as cv2
import numpy as np
from sklearn.cluster import OPTICS, DBSCAN
import os


def opening(image, kernel1, kernel2):
    image = cv2.erode(image, kernel1, 1)
    image = cv2.dilate(image, kernel2, 1)
    # image = cv2.erode(image, kernel2, 1)
    # image = cv2.dilate(image, kernel1, 1threshold_value)
    return image


def main():
    # print('Finish')

    # parameter=================================================
    # count = 0
    # outset_frame = 0
    # cycle_point = 0
    # channel = 2  # H:0 S:1 V:2
    # window_length = 10
    #
    # kernel_size1 = (5, 5)
    # kernel_size2 = (7, 7)cv2.destroyAllWindows()
    # threshold_value = 0.07
    # median_blur_value = 3
    # wait_time = 100
    # from para import para_4723_0306_150000 as para
    # count, outset_frame, cycle_point, channel, window_length, kernel_size1, kernel_size2, threshold_value, median_blur_value, wait_time, input_video = para.import_para()

    from para import para_1702_0623_0800 as para
    count, outset_frame, cycle_point, channel, window_length, kernel_size1, kernel_size2, threshold_value, median_blur_value, wait_time = para.import_para()

    video_list = [i for i in os.listdir("./video/") if (i[-3::] == "mp4")]
    for num, video_name in enumerate(video_list):
        print("{}. {}".format(num, video_name))
    input_video = "./video/" + video_list[int(input('Select video: '))]
    print(video_list[3])

    # input_video = './綠環境-(A15)家電冷氣機設備投入口及處理線、冷媒吸取區-20200331114524-20200306-130000-movie.mp4'
    # input_video = './綠環境-(A15)家電冷氣機設備投入口及處理線、冷媒吸取區-20200331114640-20200306-150000-movie.mp4'
    # input_video = './綠環境-(A15)家電冷氣機設備投入口及處理線、冷媒吸取區-20200331114723-20200306-150000-movie.mp4'
    # input_video = './output.mp4'
    # input_video = './綠環境-(B5)家電冷氣機冷媒吸取區、冷氣機及洗衣機處理線-20200331115136-20200306-110000-movie.mp4'
    # input_video = './20200416大東方-鏡頭13到鏡頭16-20200416-140000.mp4_20200423_091228.mkv'
    # input_video = './20200416大東方-鏡頭13到鏡頭16-20200416-100000.mp4_20200423_090536.mkv'
    # input_video = './5071922111_20200902_090000.mp4'
    # input_video = '/home/spie/Desktop/EPA/Test/0306/綠電楊梅-01廠區大門、02廢物品已_待認證區A、G、S、03廢物品已_待認證區H、G、S、T、04廢物品已_待認證區A、B、C廢物品已_待認證區D、F-20210312-150000.mp4'

    video_capture = cv2.VideoCapture(input_video)

    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success, frame = video_capture.read()  # 讀幀

    hsv_window = np.zeros((window_length, 480, 720, 3), np.float32)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    print(frame.shape)
    out1 = cv2.VideoWriter('outputVideo.avi', fourcc, 6.99, (frame.shape[1], frame.shape[0]))
    while success:
        print(count)

        count += 1

        if count > frame_count - 1:
            break
        success, frame = video_capture.read()  # 讀取下一幀

        if count > outset_frame:
            frame_blur = cv2.medianBlur(frame, median_blur_value)  # 中值濾波 rgb
            current_frame_hsv = cv2.cvtColor(
                frame_blur, cv2.COLOR_RGB2HSV) / 255  # 轉HSV並轉0~1
            if cycle_point >= window_length:
                cycle_point = 0
            if count < outset_frame + window_length:
                hsv_window[cycle_point] = current_frame_hsv
                cycle_point += 1
            if count == outset_frame + window_length:
                avg_value = sum(hsv_window[0:window_length, :, :, channel]) / window_length

            if count > outset_frame + window_length:
                frame_diff = abs(avg_value[:, :] - current_frame_hsv[:, :, channel])
                frame_diff = frame_diff / frame_diff.max()

                ret, frame_binary = cv2.threshold(
                    frame_diff, threshold_value, 1, cv2.THRESH_BINARY)

                img = (frame_binary * 255).astype(np.uint8)

                kernel1 = np.ones(kernel_size1, np.uint8)  # kernel
                kernel2 = np.ones(kernel_size2, np.uint8)  # kernel

                img = opening(img, kernel1, kernel2)

                clusteringDH(current_frame_hsv, img,out1,frame)
                cv2.imshow('frame', frame)
                if cv2.waitKey(wait_time) == ord('w'):
                    cv2.waitKey(0)

                avg_value = avg_value - hsv_window[cycle_point, :, :, channel] / \
                            window_length + current_frame_hsv[:, :, channel] / window_length

                # statisticsBackground(hsv_window, avg_value, current_frame_hsv, channel, kernel1, kernel2)
                hsv_window[cycle_point, :, :, channel] = current_frame_hsv[:, :, channel]
                cycle_point += 1


def clusteringDH(current_frame_HSV, img, out1, frame, sat_th=0.17):

    vimg = (current_frame_HSV[:, :, 2] * 255).astype(np.uint8)
    v2_grayimg = cv2.merge([vimg, vimg, vimg])
    result = (img * 255).astype(np.uint8)
    result1 = v2_grayimg
    contours, hierarchy = cv2.findContours(result.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v2_grayimg, contours, -1, (0, 0, 255), 2)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    x, y = np.where(result == 1)
    data = np.dstack([x, y])
    data = data[0]
    # clustering = OPTICS(min_samples=50).fit(data)
    try:
        clustering = DBSCAN(eps=5, min_samples=70).fit(data)
    except Exception as e:
        print(e)
        return
    # print(clustering.labels_)
    last = max(clustering.labels_)
    for i in range(0, last + 1):
        a = np.where(clustering.labels_ == i)
        if ((max(y[a]) - min(y[a])) * (max(x[a]) - min(x[a])) > 400) and (
                current_frame_HSV[min(x[a]):max(x[a]), min(y[a]):max(y[a]), 1].mean() < sat_th):
            # if current_frame_HSV[min(x[a]):max(x[a]), min(y[a]):max(y[a]), 1].mean() < 0.07:

            cv2.rectangle(result1, (max(y[a]), max(x[a])), (min(y[a]), min(x[a])), (0, 255, 0), 2)
            cv2.rectangle(frame, (max(y[a]), max(x[a])), (min(y[a]), min(x[a])), (0, 255, 0), 2)
            cv2.imshow('Test', result1)
            cv2.imshow('obj', current_frame_HSV[min(x[a]):max(x[a]), min(y[a]):max(y[a]), 1])
            print('mean saturation value: ', current_frame_HSV[min(x[a]):max(x[a]), min(y[a]):max(y[a]), 1].mean())
            cv2.waitKey(0)

    out1.write(frame)
    cv2.imshow('Test', result1)


def statisticsBackground(hsvWindow, avgValue, currentFrameHSV, channel, kernel1, kernel2):
    std_value = np.std(hsvWindow[:, :, :, channel], axis=0)
    std_upper = avgValue + 2 * std_value + 0.01
    std_lower = avgValue - 2 * std_value - 0.01
    moving_obj_filter = (currentFrameHSV[:, :, channel] <= std_upper) & (currentFrameHSV[:, :, channel] >= std_lower)
    # print(moving_obj_filter)
    # print(frameHSV[:, :, channel])
    # print(std_upper)
    # print(std_lower)
    now = currentFrameHSV[:, :, channel].copy()
    now[moving_obj_filter] = 0
    now[~moving_obj_filter] = 1
    now = opening(now, kernel1, kernel2)
    cv2.imshow('statisticsBackground', now)


main()
cv2.destroyAllWindows()
