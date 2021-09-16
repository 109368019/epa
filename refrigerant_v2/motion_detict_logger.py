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
import re


@nb.jit
def warmup_counting(statistics_index, window_hsv, value_counting_in_frame):
    # 統計不同亮度值的數量
    for x in range(frame_x):
        for y in range(frame_y):
            value_counting_in_frame[x, y, int(window_hsv[statistics_index, x, y, statistics_channel] * 255)] = \
                value_counting_in_frame[x, y, int(window_hsv[statistics_index, x, y, statistics_channel] * 255)] + 1
    return value_counting_in_frame


@nb.jit
def update_counting(statistics_index, window_hsv, value_counting_in_frame, current_frame_HSV):
    # x = np.take_along_axis(v_count, int(hsv[z, :, :, channel] * 255).reshape(480, 720, 1).astype(int), axis=2) + 1
    # np.put_along_axis(v_count, int(hsv[z, :, :, channel] * 255).reshape(480, 720, 1).astype(int), x, axis=2)
    # 更新統計不同亮度值的數量
    for x in range(frame_x):
        for y in range(frame_y):
            value_counting_in_frame[x, y, int(window_hsv[statistics_index, x, y, statistics_channel] * 255)] = \
                value_counting_in_frame[x, y, int(window_hsv[statistics_index, x, y, statistics_channel] * 255)] - 1
            value_counting_in_frame[x, y, int(current_frame_HSV[x, y, statistics_channel] * 255)] = \
                value_counting_in_frame[x, y, int(current_frame_HSV[x, y, statistics_channel] * 255)] + 1
    return value_counting_in_frame


@nb.jit
def and_func(previous_different_frame, different_frame):
    and_frame = ((previous_different_frame / 255 + different_frame / 255) / 2).astype('uint8')
    and_frame = and_frame * 255
    return and_frame




#  all para ==================================
all_start_time = time.time()
para = Parameter()
is_first_statuation = False
statistics_index = 0
video_dir_path = "../video/B5/"
video_list = [i for i in os.listdir(video_dir_path) if (i[-3::] == "mp4")]
video_list = sorted(video_list)

for num, video_name in enumerate(video_list):
    print(num, video_name)
    # each time para ===================================================================
    each_start_time = time.time()
    input_video_path = video_dir_path + video_list[num]
    try:
        print(video_list[num][:-11] == video_list[num - 1][:-11])
        if not video_list[num][:-11] == video_list[num - 1][:-11]:
            is_first_statuation = False
    except:
        pass
    directory_name = input_video_path[-19:-4]

    environment_name = re.findall('([^-]+)', video_list[num])[0]
    camera_name_list = re.findall('([A-Z]+[0-9]+)', re.findall('([^-]+)', video_list[num])[1]) \
        if len(re.findall('([A-Z]+[0-9]+)', re.findall('([^-]+)', video_list[num])[1])) > 0 \
        else re.findall('([0-9]+)', re.findall('([^-]+)', video_list[num])[1])
    camera_name = ""
    for name in camera_name_list:
        camera_name = camera_name + name
    video_date = re.findall('([^-]+)', video_list[num])[2]
    video_time = re.findall('([^-]+)', video_list[num])[3][:-4]

    output_path = '../ref/{}/{}/{}'.format(video_date, environment_name, camera_name)
    log_file_path = output_path + "/log/" + directory_name + ".json"
    video_red_path = output_path + "/video_red/"
    roi_csv_path = "./roi/{}.csv".format(video_list[num][:-20])
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + "/log")
        os.makedirs(output_path + "/event")
        os.makedirs(output_path + "/output_video")
        os.makedirs(output_path + "/video_red")

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
        value_counting_in_frame = np.zeros((frame_x, frame_y, 257), np.uint16)  # 統計直方圖
        previous_different_frame = np.zeros((frame_x, frame_y), np.uint8)
    and_frame = np.zeros((frame_x, frame_y), np.uint8)

    window_hsv = np.zeros((para.window_length, frame_x, frame_y, 3), np.float32)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    red_video = cv2.VideoWriter(video_red_path + directory_name + '-' + current_time + '.mp4', fourcc, 6.99,
                                (frame.shape[1], frame.shape[0]))
    log_file = {}
    # roi===========================================
    area_list = []
    with open(roi_csv_path, newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter='|')
        row = list(csvReader)
        for each_area in row:
            area_tmp = []
            for each_point in each_area:
                x, y = each_point.split(",")
                area_tmp.append([int(x), int(y)])

            area_list.append(np.array(area_tmp))
    # roi===========================================
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
            current_frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_RGB2HSV) / 255

            if statistics_index >= para.window_length:  # 統計標籤大於閥值則歸零
                statistics_index = 0

            if current_frame_index < para.start_frame + para.window_length and not is_first_statuation:
                window_hsv[statistics_index] = current_frame_HSV
                value_counting_in_frame = warmup_counting(statistics_index, window_hsv,
                                                          value_counting_in_frame)  # 統計初始值方圖
                statistics_index = statistics_index + 1

            if current_frame_index == para.start_frame + para.window_length and not is_first_statuation:
                value_counting_in_frame = warmup_counting(statistics_index, window_hsv, value_counting_in_frame)
                is_first_statuation = True

            if current_frame_index > para.start_frame + para.window_length or is_first_statuation:
                different_frame = np.zeros((frame_x, frame_y), np.uint8)  # 宣告
                moving_obj_frame_temp = different_frame[:, :]  # 宣告
                moving_obj_bool_frame = np.take_along_axis(value_counting_in_frame,
                                                           (current_frame_HSV[:, :, statistics_channel] * 255).
                                                           reshape(frame_x, frame_y, 1).astype(int),
                                                           axis=2) < para.probability_throuhold
                moving_obj_frame_temp[moving_obj_bool_frame.reshape(frame_x, frame_y)] = 255
                different_frame[:, :] = moving_obj_frame_temp
                and_frame = and_func(previous_different_frame, different_frame)
                previous_different_frame = different_frame
                img = and_frame

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
                # area1 = np.array([[0, 0], [720, 0], [720, 90], [0, 140]])
                # area2 = np.array([[0, 373], [720, 231], [720, 480], [0, 480]])
                # area_list = [area1, area2]
                # 綠環境roi===========================================

                cv2.polylines(frame, area_list, True, (0, 0, 255), 2)
                cv2.fillPoly(img, area_list, (0, 0, 0))

                # ====================
                cv2.imshow('frame', frame)
                cv2.imshow('img', img)
                cv2.waitKey(1)
                # ====================

                ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(contour_frame, ctrs, -1, (0, 0, 255), 2)

                different_frame_x, different_frame_y = np.where(img[:, :] == 255)
                different_points_coordinates = np.dstack([different_frame_x, different_frame_y])
                different_points_coordinates = different_points_coordinates[0]
                try:
                    clustering = DBSCAN(
                        eps=para.eps_value, min_samples=para.min_samples_size).fit(different_points_coordinates)

                    all_labels_array = np.setdiff1d(np.unique(clustering.labels_),
                                                    np.array([-1]))  # 分群的結果可能會有 -1 類別(無歸屬)，此類別須排除

                    each_frame_para_list = []
                    H = current_frame_HSV[:, :, 0]
                    S = current_frame_HSV[:, :, 1]
                    V = current_frame_HSV[:, :, 2]
                    total_avg_value = V.mean()
                    total_avg_sat = S.mean()

                    for each_label in all_labels_array:
                        labels_index = np.where(clustering.labels_ == each_label)
                        class_map = np.zeros((current_frame_HSV.shape[0], current_frame_HSV.shape[1]))
                        points_of_coordinates_in_label = [
                            (different_frame_x[each_point_index], different_frame_y[each_point_index]) for
                            each_point_index in labels_index]

                        for points_x, points_y in points_of_coordinates_in_label:
                            class_map[points_x, points_y] = 1

                        # criteria condition
                        class_map_num = len(points_of_coordinates_in_label[0][0])
                        class_map_criteria = class_map_num > para.class_map_num_value

                        if (class_map_criteria):
                            obj_brighten = False
                            try:
                                obj_brighten = True if window_hsv[statistics_index - 1, :, :, 2][
                                                           class_map == 1].mean() - \
                                                       window_hsv[statistics_index - 2, :, :, 2][
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

                    if len(each_frame_para_list) > 0:
                        log_file.update(
                            {current_frame_index: {"total_avg_value": total_avg_value, "total_avg_sat": total_avg_sat,
                                                   "obj_count": len(each_frame_para_list),
                                                   "obj_data": each_frame_para_list}})
                except Exception as e:
                    pass
                red_video.write(contour_frame)

                value_counting_in_frame = update_counting(statistics_index, window_hsv, value_counting_in_frame,
                                                          current_frame_HSV)  # 更新值方圖
                window_hsv[statistics_index, :, :, statistics_channel] = current_frame_HSV[:, :, statistics_channel]
                statistics_index = statistics_index + 1
    with open(log_file_path, 'w') as fp:
        json.dump(log_file, fp)
    end_time = time.time()
    execution_time = end_time - each_start_time
    print("execution time = ", execution_time)
all_end_time = time.time()
all_execution_time = all_end_time - all_start_time
print("all_execution time = ", all_execution_time)
