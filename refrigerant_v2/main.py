import os
import sys
import re
import time
import cv2
import joblib
import numpy as np
from para.parameter import Parameter
import csv
import numba as nb
from sklearn.cluster import DBSCAN
import json
import threading

#關閉warnings
import warnings
warnings.filterwarnings('ignore')

class Refrigerant:
    def __init__(self, video_dir_path: str):
        self.area_list = []
        self.video_dir_path = video_dir_path
        self.is_first_statistics = False
        self.statistics_index = 0
        self.statistics_channel = 2
        self.para = Parameter()
        self.path_dict = dict()
        self.report_list = []
        self.svm_model = joblib.load('clf3.pkl')
        self.DCT_LEVEL = 32

    def creat_path(self, input_video_name: str):
        # print(self.video_dir_path)
        self.input_video_path = self.video_dir_path + input_video_name
        self.file_name = self.input_video_path[-19:-4]
        environment_name = re.findall('([^-]+)', input_video_name)[0]
        self.report_list.append(environment_name)
        camera_name_list = re.findall('([A-Z]+[0-9]+)', re.findall('([^-]+)', input_video_name)[1]) \
            if len(re.findall('([A-Z]+[0-9]+)', re.findall('([^-]+)', input_video_name)[1])) > 0 \
            else re.findall('([0-9]+)', re.findall('([^-]+)', input_video_name)[1])
        camera_name = ""
        for name in camera_name_list:
            camera_name = camera_name + name
        self.report_list.append(camera_name)
        video_date = re.findall('([^-]+)', input_video_name)[2]
        video_time = re.findall('([^-]+)', input_video_name)[3][:-4]
        print(environment_name, camera_name, video_date, video_time)
        self.report_list.append(video_date)
        self.report_list.append(video_time[:-4])

        # output_path = '../ref/{}/{}/{}'.format(video_date, environment_name, camera_name)
        output_path = '../output/{}/{}/{}'.format(video_date, environment_name, camera_name)
        print("output path: ", output_path)
        self.roi_csv_file_path = "./roi/{}.csv".format(input_video_name[:-20])
        self.log_path = output_path + "/log/"
        self.event_path = output_path + "/event/"
        self.video_red_path = output_path + "/video_red/"
        self.output_video_path = output_path + "/output_video/"

        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            os.makedirs(output_path + "/log")
            os.makedirs(output_path + "/event")
            os.makedirs(output_path + "/output_video")
            os.makedirs(output_path + "/video_red")

        self.path_dict["input_video_path"] = self.input_video_path
        self.path_dict["roi_csv_file_path"] = self.roi_csv_file_path
        self.path_dict["log_path"] = self.log_path
        self.path_dict["event_path"] = self.event_path
        self.path_dict["video_red_path"] = self.video_red_path
        self.path_dict["output_video_path"] = self.output_video_path
        self.path_dict["file_name"] = self.file_name

        return self.path_dict

    def read_roi_file(self, roi_csv_file_path):
        print("ROI file path: ", roi_csv_file_path)
        # input()
        self.area_list = []
        with open(roi_csv_file_path, newline='') as csvFile:
            csvReader = csv.reader(csvFile, delimiter='|')
            row = list(csvReader)
            for each_area in row:
                area_tmp = []
                for each_point in each_area:
                    x, y = each_point.split(",")
                    area_tmp.append([int(x), int(y)])
                self.area_list.append(np.array(area_tmp))

    def generate_svm_feature(self, img):
        # print("generate_svm_feature")
        # cv2.imshow("df", img)
        # cv2.waitKey(1)
        pic = cv2.resize(img, (self.DCT_LEVEL, self.DCT_LEVEL), interpolation=cv2.INTER_NEAREST)
        pic_float = np.float32(pic) / 256
        pic_dct = cv2.dct(pic_float[:, :, 2])
        pic_dct_list = pic_dct.reshape(1, -1)[0]
        # print(pic_dct_list)
        return pic_dct_list

    def svm_predict(self, img_feature):
        predict_result = self.svm_model.predict([img_feature])
        return predict_result[0]

    @nb.jit
    def warmup_counting(self, statistics_index, window_hsv, value_counting_in_frame, frame_x, frame_y,
                        statistics_channel=2):
        # 統計不同亮度值的數量
        for x in range(frame_x):
            for y in range(frame_y):
                value_counting_in_frame[x, y, int(window_hsv[statistics_index, x, y, statistics_channel] * 255)] += 1
        return value_counting_in_frame

    @nb.jit
    def and_func(self, previous_different_frame, different_frame):
        and_frame = ((previous_different_frame / 255 + different_frame / 255) / 2).astype('uint8')
        and_frame = and_frame * 255
        return and_frame

    @nb.jit
    def update_counting(self, statistics_index, window_hsv, value_counting_in_frame, current_frame_HSV, frame_x,
                        frame_y, statistics_channel=2):
        # 更新統計不同亮度值的數量
        for x in range(frame_x):
            for y in range(frame_y):
                value_counting_in_frame[x, y, int(window_hsv[statistics_index, x, y, statistics_channel] * 255)] = \
                    value_counting_in_frame[
                        x, y, int(window_hsv[statistics_index, x, y, statistics_channel] * 255)] - 1
                value_counting_in_frame[x, y, int(current_frame_HSV[x, y, statistics_channel] * 255)] = \
                    value_counting_in_frame[x, y, int(current_frame_HSV[x, y, statistics_channel] * 255)] + 1
        return value_counting_in_frame

    def motion_detict_logger(self, input_video_path, output_log_path, roi_csv_file_path, file_name, save_video=True):
        videoCapture = cv2.VideoCapture(input_video_path)
        print("source video path: ", input_video_path)
        frames_num_of_input_video = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame_index = 0
        success, frame = videoCapture.read()  # 讀幀
        frame_x = frame.shape[0]  # 480
        frame_y = frame.shape[1]  # 720
        print("frame size: {}, {}".format(frame_x, frame_y))

        if not self.is_first_statistics:
            self.value_counting_in_frame = np.zeros((frame_x, frame_y, 257), np.uint16)  # 統計直方圖
            self.previous_different_frame = np.zeros((frame_x, frame_y), np.uint8)
            self.statistics_index = 0
            self.window_hsv = np.zeros((self.para.window_length, frame_x, frame_y, 3), np.float32)

        # else:
        #     self.statistics_index = self.statistics_index + 1

        current_time = time.strftime("%Y%d%m%H%M%S", time.localtime())

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        if save_video:
            red_video = cv2.VideoWriter(self.video_red_path + self.file_name + '-' + current_time + '.mp4', fourcc,
                                        6.99, (frame.shape[1], frame.shape[0]))
        log_file = {}
        self.read_roi_file(roi_csv_file_path)
        while success:

            # print("\r{}".format(current_frame_index), end="")
            # print("self.statistics_index:", self.statistics_index)
            current_frame_index = current_frame_index + 1
            print("\r[motion_detict_logger_progress] |{}{}| {:>5.0f}/{:>5.0f} - {:>3.0f}%".format(
                '█' * int(current_frame_index * 20 / frames_num_of_input_video),
                ' ' * (20 - int(current_frame_index * 20 / frames_num_of_input_video)), current_frame_index,
                frames_num_of_input_video, (current_frame_index / frames_num_of_input_video) * 100), end="")
            if current_frame_index > frames_num_of_input_video - 1:  # 若是超出影片frame數，則跳出
                break

            success, frame = videoCapture.read()  # 讀取下一幀
            contour_frame = frame.copy()

            if current_frame_index > self.para.start_frame or self.is_first_statistics:  # 開始
                frame_blur = cv2.medianBlur(frame, self.para.median_blur_value, cv2.BORDER_DEFAULT)  # 中值濾波 rgb
                frame_blur = cv2.GaussianBlur(frame_blur, self.para.gaussian_kernal_size,
                                              cv2.BORDER_DEFAULT)  # 高斯濾波 rgb
                current_frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_RGB2HSV) / 255

                if self.statistics_index >= self.para.window_length:  # 統計標籤大於閥值則歸零
                    self.statistics_index = 0

                if current_frame_index < self.para.start_frame + self.para.window_length and not self.is_first_statistics:
                    self.window_hsv[self.statistics_index] = current_frame_HSV
                    self.value_counting_in_frame = self.warmup_counting(self.statistics_index, self.window_hsv,
                                                                        self.value_counting_in_frame, frame_x,
                                                                        frame_y)  # 統計初始值方圖
                    self.statistics_index = self.statistics_index + 1

                if current_frame_index == self.para.start_frame + self.para.window_length and not self.is_first_statistics:
                    self.window_hsv[self.statistics_index] = current_frame_HSV
                    self.value_counting_in_frame = self.warmup_counting(self.statistics_index, self.window_hsv,
                                                                        self.value_counting_in_frame, frame_x, frame_y)
                    self.is_first_statistics = True

                if current_frame_index > self.para.start_frame + self.para.window_length or self.is_first_statistics:
                    different_frame = np.zeros((frame_x, frame_y), np.uint8)  # 宣告
                    moving_obj_frame_temp = different_frame[:, :]  # 宣告
                    moving_obj_bool_frame = np.take_along_axis(self.value_counting_in_frame,
                                                               (current_frame_HSV[:, :, self.statistics_channel] * 255).
                                                               reshape(frame_x, frame_y, 1).astype(int),
                                                               axis=2) < self.para.probability_throuhold
                    moving_obj_frame_temp[moving_obj_bool_frame.reshape(frame_x, frame_y)] = 255
                    different_frame[:, :] = moving_obj_frame_temp
                    and_frame = self.and_func(self.previous_different_frame, different_frame)

                    self.previous_different_frame = different_frame
                    img = and_frame

                    # 侵蝕膨脹
                    img = cv2.erode(img, None, iterations=2)  # 侵蝕膨脹去雜訊
                    img = cv2.dilate(img, None, iterations=2)

                    cv2.polylines(frame, self.area_list, True, (0, 0, 255), 2)
                    cv2.fillPoly(img, self.area_list, (0, 0, 0))

                    ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contour_frame, ctrs, -1, (0, 0, 255), 2)
                    # TODO debug====
                    # cv2.imshow('different_frame', different_frame)
                    # cv2.imshow('and_frame', and_frame)
                    # cv2.imshow('frame', frame)
                    # cv2.imshow('img', img)
                    # cv2.imshow('contour_frame', contour_frame)
                    # cv2.waitKey(1)
                    # debug====

                    different_frame_x, different_frame_y = np.where(img[:, :] == 255)
                    different_points_coordinates = np.dstack([different_frame_x, different_frame_y])
                    different_points_coordinates = different_points_coordinates[0]

                    try:
                        clustering = DBSCAN(
                            eps=self.para.eps_value, min_samples=self.para.min_samples_size).fit(
                            different_points_coordinates)

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
                            class_map_criteria = class_map_num > self.para.class_map_num_value

                            if (class_map_criteria):
                                # TODO debug====
                                # cv2.rectangle(contour_frame, (
                                #     max(different_frame_y[labels_index]), max(different_frame_x[labels_index])), (
                                #                   min(different_frame_y[labels_index]),
                                #                   min(different_frame_x[labels_index])),
                                #               (0, 255, 0), 2)
                                # debug====

                                obj_brighten = False
                                try:
                                    obj_brighten = True if current_frame_HSV[:, :, 2][
                                                               class_map == 1].mean() - \
                                                           self.window_hsv[self.statistics_index - 1, :, :, 2][
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
                                # TODO debug====
                                # print(para_dict)
                                # cv2.imshow('contour_frame', contour_frame)
                                # cv2.waitKey(0)
                                # debug====

                        if len(each_frame_para_list) > 0:
                            log_file.update(
                                {current_frame_index: {"total_avg_value": total_avg_value,
                                                       "total_avg_sat": total_avg_sat,
                                                       "obj_count": len(each_frame_para_list),
                                                       "obj_data": each_frame_para_list}})
                    except Exception as e:
                        pass
                    if save_video:
                        red_video.write(contour_frame)

                    self.value_counting_in_frame = self.update_counting(self.statistics_index, self.window_hsv,
                                                                        self.value_counting_in_frame,
                                                                        current_frame_HSV, frame_x, frame_y)  # 更新值方圖
                    self.window_hsv[self.statistics_index, :, :, self.statistics_channel] = current_frame_HSV[:, :,
                                                                                            self.statistics_channel]
                    # print("sum:", sum(self.value_counting_in_frame[0][0]))
                    self.statistics_index = self.statistics_index + 1
        with open(output_log_path + file_name + ".json", 'w') as fp:
            json.dump(log_file, fp)

    def motion_detict_logger_test(self, input_video_path, output_log_path, roi_csv_file_path, file_name,
                                  save_video=True):
        videoCapture = cv2.VideoCapture(input_video_path)
        print("source video path: ", input_video_path)
        frames_num_of_input_video = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame_index = 0
        success, frame = videoCapture.read()  # 讀幀
        frame_x = frame.shape[0]  # 480
        frame_y = frame.shape[1]  # 720
        print("frame size: {}, {}".format(frame_x, frame_y))

        if not self.is_first_statistics:
            self.value_counting_in_frame = np.zeros((frame_x, frame_y, 257), np.uint16)  # 統計直方圖
            self.previous_different_frame = np.zeros((frame_x, frame_y), np.uint8)
            self.statistics_index = 0
            self.window_hsv = np.zeros((self.para.window_length, frame_x, frame_y, 3), np.float32)

        # else:
        #     self.statistics_index = self.statistics_index + 1

        current_time = time.strftime("%Y%d%m%H%M%S", time.localtime())

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')

        if save_video:
            red_video = cv2.VideoWriter(self.video_red_path + self.file_name + '-' + current_time + '.mp4', fourcc,
                                        6.99, (frame.shape[1], frame.shape[0]))
        log_file = {}
        self.read_roi_file(roi_csv_file_path)
        while success:
            # print("\r{}".format(current_frame_index), end="")
            # print("self.statistics_index:", self.statistics_index)
            current_frame_index = current_frame_index + 1
            print("\r[motion_detict_logger_progress] |{}{}| {:>5.0f}/{:>5.0f} - {:>3.0f}%".format(
                '█' * int(current_frame_index * 20 / frames_num_of_input_video),
                ' ' * (20 - int(current_frame_index * 20 / frames_num_of_input_video)), current_frame_index,
                frames_num_of_input_video, (current_frame_index / frames_num_of_input_video) * 100), end="")
            # print("statistics_index", self.statistics_index)

            if current_frame_index > frames_num_of_input_video - 1:  # 若是超出影片frame數，則跳出
                break

            success, frame = videoCapture.read()  # 讀取下一幀
            contour_frame = frame.copy()

            if current_frame_index > self.para.start_frame or self.is_first_statistics:  # 開始
                frame_blur = cv2.medianBlur(frame, self.para.median_blur_value, cv2.BORDER_DEFAULT)  # 中值濾波 rgb
                frame_blur = cv2.GaussianBlur(frame_blur, self.para.gaussian_kernal_size,
                                              cv2.BORDER_DEFAULT)  # 高斯濾波 rgb
                current_frame_HSV = cv2.cvtColor(frame_blur, cv2.COLOR_RGB2HSV) / 255

                if self.statistics_index >= self.para.window_length:  # 統計標籤大於閥值則歸零
                    self.statistics_index = 0

                if current_frame_index < self.para.start_frame + self.para.window_length and not self.is_first_statistics:
                    # print("curr {} sta {} < ".format(current_frame_index, self.statistics_index))
                    self.window_hsv[self.statistics_index] = current_frame_HSV
                    self.value_counting_in_frame = self.warmup_counting(self.statistics_index, self.window_hsv,
                                                                        self.value_counting_in_frame, frame_x,
                                                                        frame_y)  # 統計初始值方圖
                    self.statistics_index = self.statistics_index + 1

                if current_frame_index == self.para.start_frame + self.para.window_length and not self.is_first_statistics:
                    # print("curr {} sta {} = ".format(current_frame_index, self.statistics_index))
                    self.window_hsv[self.statistics_index] = current_frame_HSV

                    self.value_counting_in_frame = self.warmup_counting(self.statistics_index, self.window_hsv,
                                                                        self.value_counting_in_frame, frame_x, frame_y)
                    self.is_first_statistics = True

                if current_frame_index > self.para.start_frame + self.para.window_length or self.is_first_statistics:
                    different_frame = np.zeros((frame_x, frame_y), np.uint8)  # 宣告
                    moving_obj_frame_temp = different_frame[:, :]  # 宣告

                    # start =time.time()
                    moving_obj_bool_frame = np.take_along_axis(
                        self.value_counting_in_frame, 
                        (current_frame_HSV[:, :, self.statistics_channel] * 255).reshape(frame_x, frame_y, 1).astype(int),
                        axis=2
                    ) < self.para.probability_throuhold
                    # print(time.time()-start)

                    moving_obj_frame_temp[moving_obj_bool_frame.reshape(frame_x, frame_y)] = 255
                    different_frame[:, :] = moving_obj_frame_temp
                    and_frame = self.and_func(self.previous_different_frame, different_frame)

                    img = and_frame

                    # 侵蝕膨脹
                    img = cv2.erode(img, None, iterations=2)  # 侵蝕膨脹去雜訊
                    img = cv2.dilate(img, None, iterations=2)

                    cv2.polylines(contour_frame, self.area_list, True, (0, 0, 255), 2)
                    cv2.fillPoly(img, self.area_list, (0, 0, 0))

                    ctrs, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(contour_frame, ctrs, -1, (0, 0, 255), 2)
                    # TODO debug====
                    # cv2.imshow('different_frame', different_frame)
                    # cv2.imshow('and_frame', and_frame)
                    # cv2.imshow('frame', frame)
                    # cv2.imshow('img', img)

                    # cv2.imshow('self.previous_different_frame', self.previous_different_frame)
                    # cv2.imshow('different_frame', different_frame)
                    # cv2.imshow("curr", current_frame_HSV[:, :, 2])
                    # cv2.imshow('diff-1', self.window_hsv[self.statistics_index - 1, :, :, 2])
                    # cv2.imshow('contour_frame', contour_frame)
                    # cv2.waitKey(1)
                    # debug====

                    different_frame_x, different_frame_y = np.where(img[:, :] == 255)
                    different_points_coordinates = np.dstack([different_frame_x, different_frame_y])
                    different_points_coordinates = different_points_coordinates[0]
                    # print(different_points_coordinates)
                    if(len(different_points_coordinates)!=0):
                        # TODO try 8-connect

                        # clustering = DBSCAN(
                        #     eps=self.para.eps_value, min_samples=self.para.min_samples_size).fit(
                        #     different_points_coordinates)

                        # all_labels_array = np.setdiff1d(np.unique(clustering.labels_),
                        #                                 np.array([-1]))  # 分群的結果可能會有 -1 類別(無歸屬)，此類別須排除

                        # each_frame_para_list = []
                        # H = current_frame_HSV[:, :, 0]
                        # S = current_frame_HSV[:, :, 1]
                        # V = current_frame_HSV[:, :, 2]
                        # total_avg_value = V.mean()
                        # total_avg_sat = S.mean()

                        # self.cal_para(all_labels_array, clustering, contour_frame, current_frame_HSV,
                        #               current_frame_index, different_frame_x, different_frame_y, each_frame_para_list,
                        #               frame, log_file, total_avg_sat, total_avg_value)
                        post_frame = self.window_hsv[self.statistics_index - 1, :, :, 2]
                        threading.Thread(target=self.cal_para,
                                         args=(different_points_coordinates, contour_frame, current_frame_HSV, current_frame_index, different_frame_x, different_frame_y, frame, log_file, post_frame)).start()
                    else:
                        pass

                    if save_video:
                        red_video.write(contour_frame)

                    self.value_counting_in_frame = self.update_counting(self.statistics_index, self.window_hsv,
                                                                        self.value_counting_in_frame,
                                                                        current_frame_HSV, frame_x, frame_y)  # 更新值方圖
                    self.window_hsv[self.statistics_index, :, :, self.statistics_channel] = current_frame_HSV[:, :,
                                                                                            self.statistics_channel]
                    # print("sum:", sum(self.value_counting_in_frame[0][0]))
                    self.previous_different_frame = different_frame
                    self.statistics_index = self.statistics_index + 1
                    # print("in")
        with open(output_log_path + file_name + ".json", 'w') as fp:
            json.dump(log_file, fp)

    def cal_para(self, different_points_coordinates, contour_frame, current_frame_HSV, current_frame_index, different_frame_x, different_frame_y, frame, log_file, post_frame):
        clustering = DBSCAN(
            eps=self.para.eps_value, min_samples=self.para.min_samples_size).fit(
            different_points_coordinates)

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
            class_map_criteria = class_map_num > self.para.class_map_num_value
            # TODO Thread

            if (class_map_criteria):
                obj_brighten = False
                # try:
                if((current_frame_HSV[:, :, 2][class_map == 1].mean() - 
                    post_frame[class_map == 1].mean()) > 0):
                    obj_brighten = True
                # except:
                #     obj_brighten = False
                # print(self.statistics_index)
                obj_avg_value = current_frame_HSV[:, :, 2][class_map == 1].mean()
                obj_avg_saturation = current_frame_HSV[:, :, 1][class_map == 1].mean()
                obj_std_value = current_frame_HSV[:, :, 2][class_map == 1].std()
                obj_std_saturation = current_frame_HSV[:, :, 1][class_map == 1].std()
                obj_min_x = min(points_of_coordinates_in_label[0][0])
                obj_max_x = max(points_of_coordinates_in_label[0][0])
                obj_min_y = min(points_of_coordinates_in_label[0][1])
                obj_max_y = max(points_of_coordinates_in_label[0][1])
                svm_feature = self.generate_svm_feature(frame[obj_min_x:obj_max_x, obj_min_y:obj_max_y])
                para_dict = {"obj_min_x": int(obj_min_x), "obj_max_x": int(obj_max_x),
                             "obj_min_y": int(obj_min_y), "obj_max_y": int(obj_max_y),
                             "obj_avg_value": obj_avg_value, "obj_avg_sat": obj_avg_saturation,
                             "obj_std_value": obj_std_value, "obj_std_sat": obj_std_saturation,
                             "obj_brighten": obj_brighten, "svm_predict": self.svm_predict(svm_feature)}
                each_frame_para_list.append(para_dict)
                if obj_avg_value > self.para.mean_value_limit and obj_avg_saturation < self.para.mean_satuation_limit and obj_brighten:
                    # cv2.imwrite('pic/pic_%d_%d.png' % (current_frame_index, each_label),
                    #             frame[obj_min_x:obj_max_x,
                    #             obj_min_y:obj_max_y])  # count是當前第幾幀， i是當前幀數的第幾個框
                    cv2.rectangle(contour_frame, (
                        max(different_frame_y[labels_index]), max(different_frame_x[labels_index])), (
                                      min(different_frame_y[labels_index]),
                                      min(different_frame_x[labels_index])),
                                  (0, 255, 0), 2)

                    # cv2.imwrite('pic_all/pic_%d_%d.png' % (current_frame_index, each_label),
                    #             contour_frame)  # count是當前第幾幀， i是當前幀數的第幾個框
                # self.para_calculate(class_map, contour_frame, current_frame_HSV, different_frame_x,
                #                     different_frame_y, each_frame_para_list, frame, labels_index,
                #                     points_of_coordinates_in_label)
                # threading.Thread(target=self.para_calculate, args=(class_map, contour_frame, current_frame_HSV, different_frame_x,
                #                     different_frame_y, each_frame_para_list, frame, labels_index,
                #                     points_of_coordinates_in_label)).start()

                # TODO debug====
                # print(para_dict)
                # cv2.imshow('contour_frame', contour_frame)
                # cv2.imshow("curr", current_frame_HSV[obj_min_x:obj_max_x, obj_min_y:obj_max_y, 2])
                # cv2.imshow('prev-1', self.window_hsv[self.statistics_index - 10, obj_min_x:obj_max_x,
                #                      obj_min_y:obj_max_y, 2])
                # cv2.imshow('diff-1', current_frame_HSV[obj_min_x:obj_max_x, obj_min_y:obj_max_y,
                #                      2] - self.window_hsv[self.statistics_index - 10,
                #                           obj_min_x:obj_max_x,
                #                           obj_min_y:obj_max_y, 2])
                #
                # cv2.imshow('abs-diff-1', abs(current_frame_HSV[obj_min_x:obj_max_x, obj_min_y:obj_max_y,
                #                              2] - self.window_hsv[self.statistics_index - 10,
                #                                   obj_min_x:obj_max_x,
                #                                   obj_min_y:obj_max_y, 2]))
                # print(sum(sum(abs(current_frame_HSV[obj_min_x:obj_max_x, obj_min_y:obj_max_y,
                #                   2] - self.window_hsv[self.statistics_index - 10,
                #                        obj_min_x:obj_max_x,
                #                        obj_min_y:obj_max_y, 2]))))
                # img_dct = cv2.dct(self.window_hsv[self.statistics_index - 10, obj_min_x:obj_max_x,
                #                   obj_min_y:obj_max_y, 2])

                # print("dct", img_dct)
                # cv2.imshow("dct",img_dct)
                # cv2.waitKey(1)
                # debug====

        if len(each_frame_para_list) > 0:
            log_file.update(
                {current_frame_index: {"total_avg_value": total_avg_value,
                                       "total_avg_sat": total_avg_sat,
                                       "obj_count": len(each_frame_para_list),
                                       "obj_data": each_frame_para_list}})

    # def para_calculate(self, class_map, contour_frame, current_frame_HSV, different_frame_x, different_frame_y,
    #                    each_frame_para_list, frame, labels_index, points_of_coordinates_in_label):


    def creat_event(self, log_path, event_path, file_name):
        with open(log_path + file_name + ".json", 'r') as fp:
            log = json.load(fp)

        if len(log.keys()) > 0:
            last_num = int(list(log.keys())[-1])
            first_num = int(list(log.keys())[0])
        else:
            with open(event_path + file_name + ".csv", 'w', newline='') as csvfile:
                # print("no event")
                writer = csv.writer(csvfile)
            return

        # print(first_num)
        # print(last_num)
        #讀log將所有事件分為紅框與綠框並標記該座標條件list
        #判斷事件條件與物件條件是否有交集
        #若有交集則延續事件，若有交集的物件為紅框活躍度-1並繼續追蹤，若為綠框則將活躍度重置
        #若事件與紅綠框都沒交集則結束事件
        #根據最後的活躍度減去事件幀數
        #若事件長度小於MINI_LENGTH則刪除該事件
        #若紅框沒有與任何事件交集則忽略
        #若綠框沒有與任何事件交集則新增事件

        MINI_LENGTH = 2
        stop_event_buffer = 7
        non_event_count = 0

        event_real = []  # [[start_frame,end_frame],[]...]
        event_temp = []  # [[start_frame_num, non_event_count, [present condition list],[next condition list],End_flag],...] 當 non_event_count >= 2 pop to event_real
        condition = 1
        roi_x = 240      # 需要改成讀取影片長寬
        roi_y = 360
        scope = 20       # 將影像量化成幾格
        scope_x = roi_x // scope
        scope_y = roi_y // scope
        print(scope_x,scope_y)
        for frame_num in range(first_num, last_num + 1):
            frame_data = log.get(str(frame_num), "None")
            #進度條
            print("\r[Creat_Event] |{}{}| {:>5.0f}/{:>5.0f} - {:>3.0f}%".format(
                '█' * int((frame_num - first_num + 1) * 20 / (last_num + 1 - first_num)),
                ' ' * (20 - int((frame_num - first_num + 1) * 20 / (last_num + 1 - first_num))), frame_num,
                last_num, ((frame_num - first_num) / (last_num + 1 - first_num)) * 100), end="")
            if frame_data != "None":
                # 讀取動態物
                obj_data = frame_data.get("obj_data", "None")
                new_obj_data = []                  
                # common condition
                for i in range(0, len(obj_data)):
                    #將紅框與綠框標記
                    #並將其座標清單紀錄
                    obj_loc = []
                    data_temp = obj_data[i]
                    value_criteria = data_temp["obj_avg_value"] > self.para.mean_value_limit
                    sat_criteria = data_temp["obj_avg_sat"] < self.para.mean_satuation_limit
                    # svm_criteria = data_temp["svm_predict"] == "ref"
                    for i in range((data_temp['obj_min_x']//scope_x)-1,(data_temp['obj_max_x']//scope_x)+1):
                       for j in range((data_temp['obj_min_y']//scope_y)-1,(data_temp['obj_max_y']//scope_y)+1): 
                           obj_loc.append(str(i) + ',' + str(j))
                    if (value_criteria and sat_criteria and data_temp["obj_brighten"] ):# and svm_criteria
                        ref_tag = True
                    else:
                        ref_tag = False
                    data_temp.update({"ref_tag":ref_tag})
                    data_temp.update({"obj_loc":obj_loc})
                    data_temp.update({"use_or_not":False})
                    new_obj_data.append(data_temp)

                for i in range(0, len(event_temp)): #若沒有以存在的暫存事件則跳過
                    event_loc = event_temp[i][2]
                    event_flag = False      
                    next_condition_save = []
                    for j in range(0, len(new_obj_data)):
                        
                        obj_loc = new_obj_data[j]["obj_loc"]    #讀取動態物座標清單
                        ref_tag = new_obj_data[j]["ref_tag"]    #讀取紅框綠框標記
                        
                        if len(set(event_loc) & set(obj_loc)) >= 1: #若為同事件之動態物
                            loc_condition = True
                        else:                                       #反之
                            loc_condition = False
                        
                        if loc_condition:                           
                            new_obj_data[j]["use_or_not"] = True    #將動態物標記為包含再某事件內
                            next_condition_save = list(set(next_condition_save) | set(obj_loc)) #新增下幀座標清單 （動態物聯集）
                            if ref_tag:
                                event_flag = True                   #該動態物是否為綠框
                    
                    event_temp[i][3]= next_condition_save

                    if event_flag == True:
                        event_temp[i][1] = 0
                    else:
                        event_temp[i][1] += 1
                # if new obj_event add event_temp
                # 有新物件需要新增事件
                for i in range(0, len(new_obj_data)):
                    now_condition_save = []
                    next_condition_save = []
                    now_data = new_obj_data[i]
                    if (now_data["ref_tag"]==True) and (now_data["use_or_not"]== False):
                        next_condition_save = now_data["obj_loc"]
                        event_temp.append([frame_num, 0, now_condition_save,next_condition_save,False]) 

                        # save_event = new_obj_data[i]
                #將 下幀的條件載入 並清空
                for i in range(0, len(event_temp)):
                    event_temp[i][2]=event_temp[i][3]
                    event_temp[i][3]=[]
            #若無動態物則全部事件非活躍+1
            else:
                for i in range(0, len(event_temp)):
                    event_temp[i][1] += 1
                    event_temp[i][2]=event_temp[i][3]
                    event_temp[i][3]=[]
            j=0
            #判斷事件是否結束、是否符合條件。
            for i in range(0, len(event_temp)):
                if event_temp[i-j][1] >= stop_event_buffer:
                    event_len = frame_num - event_temp[i-j][0] + 1 - stop_event_buffer
                    if event_len > MINI_LENGTH:
                        event_real.append([event_temp[i-j][0],frame_num-stop_event_buffer])
                    del event_temp[i-j]
                    j += 1
        #寫入檔案
        with open(event_path + file_name + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for i in event_real:
                writer.writerow(i)

    def creat_event_video_result(self, input_video_path, log_path, event_path, output_video_path, file_name,
                                 save_video=True):
        with open(log_path + file_name + ".json", 'r') as fp:
            log = json.load(fp)

        with open(event_path + file_name + ".csv", newline='') as csvFile:
            csvReader = csv.reader(csvFile)
            event_list = list(csvReader)
        self.report_list.append(len(event_list))

        # print("{}".format(self.file_name))
        count_event_frame = 0
        videoCapture = cv2.VideoCapture(input_video_path)
        success, frame = videoCapture.read()

        if len(event_list) == 0:  # 沒有事件
            with open(output_video_path + file_name + "_no_event" + ".csv", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
            self.report_list.append(0)
        else:
            if save_video:
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                result_video_writer = cv2.VideoWriter(
                    output_video_path + file_name + "_" + str(len(event_list)) + '.avi',
                    fourcc, 6.99,
                    (frame.shape[1], frame.shape[0]))
            for i in range(0, len(event_list)):
                # print(len(event_list),i)
                print("\r[motion_detict_logger_progress] |{}{}| {:>5.0f}/{:>5.0f} - {:>3.0f}%".format(
                    '█' * int((i + 1) * 20 / len(event_list)),
                    ' ' * (20 - int((i + 1) * 20 / len(event_list))), (i + 1),
                    len(event_list), ((i + 1) / len(event_list)) * 100), end="")

                # print("\r[creat_event_video_result] |{}{}| {:>5.0f}/{:>5.0f} - {:>3.0f}%".format(
                #     '█' * int(i+1 * 20 / len(event_list)),
                #     ' ' * (20 - int(i+1 * 20 / len(event_list))), i+1,
                #     len(event_list), (i+1 / len(event_list)) * 100), end="")
                # print("event_num", i)
                start_frame = int(event_list[i][0])
                end_frame = int(event_list[i][1])
                videoCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                frame_num = start_frame
                # windowsName = "event" + str(i)
                while (frame_num <= end_frame):
                    success, frame = videoCapture.read()
                    count_event_frame += 1
                    frame_data = log.get(str(frame_num), "None")
                    if frame_data != "None":
                        obj_data = frame_data.get("obj_data", "None")
                        new_obj_data = []
                        # common condition
                        for j in range(0, len(obj_data)):
                            data_temp = obj_data[j]
                            value_criteria = data_temp["obj_avg_value"] > self.para.mean_value_limit
                            sat_criteria = data_temp["obj_avg_sat"] < self.para.mean_satuation_limit
                            if (value_criteria and sat_criteria):
                                new_obj_data.append(data_temp)
                        for j in range(0, len(new_obj_data)):
                            data_temp = new_obj_data[j]
                            cv2.rectangle(frame, (int(data_temp["obj_max_y"]), int(data_temp["obj_max_x"])),
                                          (int(data_temp["obj_min_y"]), int(data_temp["obj_min_x"])), (0, 255, 0), 2)
                    space = len(str(i))
                    cv2.putText(frame, str(i), (360 - (space * 20), 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
                                cv2.LINE_AA)
                    if save_video:
                        result_video_writer.write(frame)
                    frame_num += 1
            self.report_list.append(round(count_event_frame / 6.99, 2))

    def save_report(self, data: list):
        # with open(event_path + file_name + ".csv", 'w', newline='') as csvfile:
        with open("report.csv", 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)

    def execute(self):
        video_list = [i for i in os.listdir(self.video_dir_path) if (i[-3::] == "mp4")]
        video_list = sorted(video_list)
        report_title_list = ["environment", "camera", "date", "time", "detected_num", "event_video_length", "run_time","end_time"]
        self.save_report(report_title_list)

        for num, video_name in enumerate(video_list):
            start_time = time.time()
            self.report_list = []
            print("[1] Creat path")
            path_dict = self.creat_path(video_name)
            # print("[2] Motion detict logger start")
            # self.motion_detict_logger_test(path_dict["input_video_path"], path_dict["log_path"],
            #                                self.path_dict["roi_csv_file_path"], path_dict["file_name"], False)
            # end_time = time.time()
            # print(" time: {}s".format(round(end_time - start_time, 2)), end="")
            # print()

            print("[3] Creat event")
            self.creat_event(path_dict["log_path"], path_dict["event_path"], path_dict["file_name"])
            print()

            # print("[4] Creat event video ref")
            # self.creat_event_video_result(path_dict["input_video_path"], path_dict["log_path"], path_dict["event_path"],
            #                               path_dict["output_video_path"], path_dict["file_name"])
            # print()

            total_end_time = time.time()
            self.report_list.append(round(total_end_time - start_time, 3))
            self.report_list.append(time.strftime("%Y/%m/%d-%H:%M:%S", time.localtime()))
            self.save_report(self.report_list)


#
# time = "20210705"
# env = "綠環境"
# cam = "B2"
# path = "../ref/{}/{}/{}/".format(time, env, cam)

video_dir_path = "../video/"
# video_dir_path = "../video/test/"
refrigerant_system = Refrigerant(video_dir_path)
refrigerant_system.execute()
refrigerant_system.save_report([])
# log_list = [i for i in os.listdir(path + "log/") if (i[-4::] == "json")]
# log_list = sorted(log_list)
# for file_name in log_list:
#     print(file_name[:-5])
#     refrigerant_system.creat_event(path + "log/", path + "event/", file_name[:-5])
# refrigerant_system.creat_event_video_result(video_dir_path, path + "log/", path + "event/", path + "output_video/",
#                                             file_name[:-5])
