# coding=UTF-8
import sys

from para.parameter import Parameter
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import numba as nb
import json
from sklearn.cluster import OPTICS, DBSCAN
import time
import csv


# videoCapture = cv2.VideoCapture('../video/2.mp4')


def opening(image, kernel):
    image = cv2.erode(image, kernel, 1)
    image = cv2.dilate(image, kernel, 1)
    return image


@nb.jit
def counting(z, hsv, v_count):
    # 統計不同亮度值的數量
    for x in range(frame_x):
        for y in range(frame_y):
            v_count[x, y, int(hsv[z, x, y, statistics_channel] * 255)] = v_count[x,
                                                                                 y, int(
                hsv[z, x, y, statistics_channel] * 255)] + 1
    return v_count


@nb.jit
def up_counting(z, hsv, v_count, frameHSV):
    # x = np.take_along_axis(v_count, int(hsv[z, :, :, channel] * 255).reshape(480, 720, 1).astype(int), axis=2) + 1
    # np.put_along_axis(v_count, int(hsv[z, :, :, channel] * 255).reshape(480, 720, 1).astype(int), x, axis=2)
    # 更新統計不同亮度值的數量
    for x in range(frame_x):
        for y in range(frame_y):
            v_count[x, y, int(hsv[z, x, y, statistics_channel] * 255)] = v_count[x,
                                                                                 y, int(
                hsv[z, x, y, statistics_channel] * 255)] - 1
            v_count[x, y, int(frameHSV[x, y, statistics_channel] * 255)] = v_count[x,
                                                                                   y, int(
                frameHSV[x, y, statistics_channel] * 255)] + 1
    return v_count


def check_mean(img, channel, x1, x2, y1, y2):
    block = img[x1:x2, y1:y2, channel]
    mean = block.mean()
    # cv2.imshow("block", block)
    print("{} mean: {}".format(channel, mean))


@nb.jit
def and_frame_func(save_frameDiff2, frameDiff2):
    # print_img = np.zeros((frame_x, frame_y), np.uint8)
    print_img = ((save_frameDiff2 / 255 + frameDiff2 / 255) / 2).astype('uint8')
    print_img = print_img * 255
    return print_img


all_start_time = time.time()

# 0814
# path = "../epa/video/"
# path = "../video/0625/"
path = "../video/0705_genv_A15/"

#  all para ==================================
# save_log_path = "./log/"
para = Parameter()
is_first_statuation = False
statistics_index = 0
# all para ==================================
video_list = [i for i in os.listdir(path) if (i[-3::] == "mp4")]
video_list = sorted(video_list)
for num, video_name in enumerate(video_list):
    print(num, video_name)
    # each para ===================================================================
    each_start_time = time.time()
    input_video_path = path + video_list[num]
    directory_name = input_video_path[-19:-4]
    output_video_path = './output_video/outputVideo_{}'.format(directory_name)

    # if not os.path.isdir(video_path):
    #     os.mkdir(video_path)

    current_time = time.strftime("%Y%d%m%H%M%S", time.localtime())
    videoCapture = cv2.VideoCapture(input_video_path)
    frames_num_of_input_video = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame_index = 0
    success, frame = videoCapture.read()  # 讀幀
    frame_x = frame.shape[0]  # 480
    frame_y = frame.shape[1]  # 720
    print(frame_x, frame_y)
    statistics_channel = 2

    if not is_first_statuation:
        v_count = np.zeros((frame_x, frame_y, 257), np.uint16)  # 統計直方圖
        previous_different_frame = np.zeros((frame_x, frame_y), np.uint8)
    and_frame = np.zeros((frame_x, frame_y), np.uint8)

    hsv = np.zeros((para.window_length, frame_x, frame_y, 3), np.float32)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out1 = cv2.VideoWriter(output_video_path + '/' + directory_name + '-' + current_time + '.avi', fourcc, 6.99,
                           (frame.shape[1], frame.shape[0]))
    log_file = {}
    # each para ===================================================================

    while success:
        print(current_frame_index)
        current_frame_index = current_frame_index + 1

        if current_frame_index > frames_num_of_input_video - 1:  # 若是超出影片frame數，則跳出
            break

        success, frame = videoCapture.read()  # 讀取下一幀
        contour_frame = frame.copy()
        clean_frame = frame.copy()

        if current_frame_index > para.start_frame or is_first_statuation:  # 開始
            frame_blur = cv2.medianBlur(frame, para.median_blur_value, cv2.BORDER_DEFAULT)  # 中值濾波 rgb
            frame_blur = cv2.GaussianBlur(frame_blur, para.gaussian_kernal_size, cv2.BORDER_DEFAULT)  # 高斯濾波 rgb
            frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_RGB2HSV) / 255

            if statistics_index >= para.window_length:  # 統計標籤大於閥值則歸零
                statistics_index = 0

            if current_frame_index < para.start_frame + para.window_length and not is_first_statuation:
                hsv[statistics_index] = frame_HSV
                v_count = counting(statistics_index, hsv, v_count)  # 統計初始值方圖
                statistics_index = statistics_index + 1

            if current_frame_index == para.start_frame + para.window_length and not is_first_statuation:
                v_count = counting(statistics_index, hsv, v_count)
                is_first_statuation = True

            if current_frame_index > para.start_frame + para.window_length or is_first_statuation:
                different_frame = np.zeros((frame_x, frame_y), np.uint8)  # 宣告
                moving_obj_frame_temp = different_frame[:, :]  # 宣告

                moving_obj_bool_frame = np.take_along_axis(v_count, (frame_HSV[:, :, statistics_channel] * 255).
                                                           reshape(frame_x, frame_y, 1).astype(int),
                                                           axis=2) < para.probability_throuhold
                moving_obj_frame_temp[moving_obj_bool_frame.reshape(frame_x, frame_y)] = 255
                different_frame[:, :] = moving_obj_frame_temp
                # 做AND
                # img = different_frame_binary

                and_frame = and_frame_func(previous_different_frame, different_frame)
                previous_different_frame = different_frame
                img = and_frame
                # cv2.imshow('and', img)

                # 侵蝕膨脹
                img = cv2.erode(img, None, iterations=2)  # 侵蝕膨脹去雜訊
                img = cv2.dilate(img, None, iterations=2)

                # ROI
                # 綠電roi===========================================
                # area1 = np.array([[0, 0], [80, 0], [0, 120]])
                # area2 = np.array([[625, 0], [720, 0], [720, 100]])
                # area3 = np.array([[366, 244], [472, 244], [442, 480], [366, 480]])
                # area4 = np.array([[610, 244], [720, 244], [720, 352]])
                # area_list = [area1, area2, area3, area4]
                # 綠電roi===========================================

                # 綠環境roi===========================================
                area1 = np.array([[0, 0], [720, 0], [720, 90], [0, 140]])
                area2 = np.array([[0, 373], [720, 231], [720, 480], [0, 480]])
                area_list = [area1, area2]
                # 綠環境roi===========================================

                cv2.polylines(frame, area_list, True, (0, 0, 255), 5)
                cv2.fillPoly(img, area_list, (0, 0, 0))

                # ====================
                cv2.imshow('frame', frame)
                cv2.imshow('img', img)
                cv2.waitKey(1)
                # ====================

                ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_frame, ctrs, -1, (0, 0, 255), 2)

                '''
                boxes = []
                for ctr in ctrs:
                    x, y, w, h = cv2.boundingRect(ctr)
                    boxes.append([x, y, w, h])  # 每個輪廓的範圍存進方形的bounding box
    
                for i, box in enumerate(boxes):
                    top_left = (box[0], box[1])
                    bottom_right = (box[0] + box[2], box[1] + box[3])
                    cv2.rectangle(frameBlur, top_left, bottom_right, (0, 255, 0), 2)  # 畫bounding box在彩色畫面上
                    # ROI_img = frame[top_left[1]:bottom_right[1],
                    #           top_left[0]:bottom_right[0]]  # 把bounding box的ROI範圍截下存成新圖(阿杜說可當訓練樣本)
                    # # cv2.imwrite('data1_2/data_%d_%d.png' %(count , i), ROI_img)     #count是當前第幾幀， i是當前幀數的第幾個框
                '''
                # ====================================================================

                different_frame_x, different_frame_y = np.where(img[:, :] == 255)
                different_points_coordinates = np.dstack([different_frame_x, different_frame_y])
                different_points_coordinates = different_points_coordinates[0]
                try:
                    clustering = DBSCAN(
                        eps=para.eps_value, min_samples=para.min_samples_size).fit(different_points_coordinates)

                    all_labels_array = np.setdiff1d(np.unique(clustering.labels_),
                                                    np.array([-1]))  # 分群的結果可能會有 -1 類別(無歸屬)，此類別須排除

                    each_frame_para_list = []
                    H = frame_HSV[:, :, 0]
                    S = frame_HSV[:, :, 1]
                    V = frame_HSV[:, :, 2]
                    total_avg_value = V.mean()
                    total_avg_sat = S.mean()

                    for each_label in all_labels_array:
                        labels_index = np.where(clustering.labels_ == each_label)
                        class_map = np.zeros((frame_HSV.shape[0], frame_HSV.shape[1]))
                        points_of_coordinates_in_label = [
                            (different_frame_x[each_point_index], different_frame_y[each_point_index]) for
                            each_point_index in labels_index]

                        for points_x, points_y in points_of_coordinates_in_label:
                            class_map[points_x, points_y] = 1

                        class_map_num = len(points_of_coordinates_in_label[0][0])
                        # ratio_value = float(
                        #    class_map_num) / ((max(xy[0][0]) - min(xy[0][0])) * (max(xy[0][1]) - min(xy[0][1])))
                        # print("ratio_value", ratio_value)
                        # ratio_criteria = ratio_value > para.ratio_limit
                        # print("ratio_criteria", ratio_criteria)
                        class_map_criteria = class_map_num > para.class_map_num_value

                        # mean_value_criteria = obj_avg_value > para.mean_value_limit
                        # mean_satuation_criteria = obj_avg_sat < para.mean_satuation_limit

                        # print(
                        #    "mean_satuation_criteria:{}, mean_value_criteria:{}, class_map_criteria:{}, ratio_criteria:{} ".format(
                        #        mean_satuation_criteria, mean_value_criteria, class_map_criteria, ratio_criteria))
                        # and mean_satuation_criteria and mean_value_criteria and ratio_criteria):

                        if (class_map_criteria):
                            obj_brighten = False
                            try:
                                obj_brighten = True if hsv[statistics_index - 1, :, :, 2][class_map == 1].mean() - \
                                                       hsv[statistics_index - 2, :, :, 2][
                                                           class_map == 1].mean() > 0 else False
                            except:
                                obj_brighten = False

                            obj_avg_value = V[class_map == 1].mean()
                            obj_avg_saturation = S[class_map == 1].mean()
                            obj_std_value = V[class_map == 1].std()
                            obj_std_saturation = S[class_map == 1].std()
                            obj_min_x = min(points_of_coordinates_in_label[0][0])
                            obj_max_x = max(points_of_coordinates_in_label[0][0])
                            obj_min_y = min(points_of_coordinates_in_label[0][1])
                            obj_max_y = max(points_of_coordinates_in_label[0][1])
                            para_dict = {"obj_min_x": int(obj_min_x), "obj_max_x": int(obj_max_x),
                                         "obj_min_y": int(obj_min_y), "obj_max_y": int(obj_max_y),
                                         "obj_avg_value": obj_avg_value, "obj_avg_sat": obj_avg_saturation,
                                         "obj_std_value": obj_std_value, "obj_std_sat": obj_std_saturation,
                                         "obj_brighten": obj_brighten}
                            each_frame_para_list.append(para_dict)
                            # LT = min(xy[0][0]

                            # contour_frame[class_map == 1] = 0
                            # cv2.imshow('obj', frameHSV[min(x[a]):max(x[a]), min(y[a]):max(y[a]), 1])
                            # print('mean saturation value: ', H[class_map == 1].mean(), S[class_map == 1].mean(),
                            #   V[class_map == 1].mean())
                            # cv2.rectangle(contour_frame, (max(y[a]), max(
                            # x[a])), (min(y[a]), min(x[a])), (0, 255, 0), 2)
                            # cv2.imwrite(img_path + '/data_%d_1.png' % (count), contour_frame)
                            # cv2.imwrite(img_path + '/data_%d_2.png' % (count), clean_frame)

                            # debug
                            # cv2.destroyWindow("obj")
                            # cv2.imshow('rect', clean_frame)
                            # cv2.imshow('Test', contour_frame)
                            # cv2.waitKey(0)
                            # csv_list = [str(count), str(min(xy[0][0]))+'_'+str(min(xy[0][1])), str(max(xy[0][0]))+'_'+str(max(xy[0][1])), str(total_avg_value)[:str(total_avg_value).find('.')+4],
                            # str(total_avg_sat)[:str(total_avg_sat).find('.')+4], str(obj_avg_value)[:str(obj_avg_value).find('.')+4], str(obj_avg_sat)[:str(obj_avg_sat).find('.')+4],
                            # str(obj_std_value)[:str(obj_std_value).find('.')+4], str(obj_std_sat)[:str(obj_std_sat).find('.')+4], 'around_avg_value', 'around_avg_sat']
                            # with open(csv_filename, 'a') as f:

                            #     writer = csv.writer(f)
                            #     writer.writerow(csv_list)
                            # cv2.imshow('Test', frame)
                    # -------------------------
                    if len(each_frame_para_list) > 0:
                        # print(data_list)
                        # print(log)

                        log_file.update(
                            {current_frame_index: {"total_avg_value": total_avg_value, "total_avg_sat": total_avg_sat,
                                                   "obj_count": len(each_frame_para_list),
                                                   "obj_data": each_frame_para_list}})
                        # print(log)
                        # with open('data1.json', 'w') as fp:
                        #     json.dump(log, fp)  

                except Exception as e:
                    pass

                # ====================================================================

                # check_mean(frameHSV, channel, 121, 209, 365, 421)

                # out1.write(contour_frame)

                # plt.figure(figsize=(15, 10), dpi=100, linewidth=2)
                # plt.plot(v_count[121, 365, :], 's-', color='r')
                # plt.show()
                # cv2.imshow('frameBlur', frameBlur)
                # cv2.imshow('dilate', img)

                # debug

                # cv2.imshow('frame', contour_frame)
                # cv2.waitKey(1)

                v_count = up_counting(statistics_index, hsv, v_count, frame_HSV)  # 更新值方圖
                hsv[statistics_index, :, :, statistics_channel] = frame_HSV[:, :, statistics_channel]
                statistics_index = statistics_index + 1

    json_Name = "./log/" + directory_name + ".json"
    with open(json_Name, 'w') as fp:
        json.dump(log_file, fp)
    end_time = time.time()
    execution_time = end_time - each_start_time
    print("execution time = ", execution_time)
    # break
# auto====================
all_end_time = time.time()
all_execution_time = all_end_time - all_start_time
print("all_execution time = ", all_execution_time)
