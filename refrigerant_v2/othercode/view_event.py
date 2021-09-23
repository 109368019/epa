import cv2 as cv2

import numpy as np
import json as json
import csv

import os

date = "0625/"
eventpath = "./event/" + date
log_path = "./log/" + date
video_path = "../video/" + date
output_path = "./output_video/" + date
if not os.path.isdir(output_path):
    os.mkdir(output_path)

video_list = [i for i in os.listdir(video_path) if (i[-3::] == "mp4")]
video_list = sorted(video_list)

for num, video_name in enumerate(video_list):
    output_name = video_name[-19:-4]

    with open(log_path + output_name + ".json", 'r') as fp:
        log = json.load(fp)

    with open(eventpath + output_name + ".csv", newline='') as csvFile:
        csvReader = csv.reader(csvFile)
        event_list = list(csvReader)

    print("{}. {}".format(num, video_name))
    input_video = video_path + video_list[num]
    videoCapture = cv2.VideoCapture(input_video)
    success, frame = videoCapture.read()
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    result_video_writer = cv2.VideoWriter(output_path + output_name + '.avi', fourcc, 6.99,
                                          (frame.shape[1], frame.shape[0]))

    for i in range(0, len(event_list)):
        start_frame = int(event_list[i][0])
        end_frame = int(event_list[i][1])
        videoCapture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_num = start_frame
        windowsName = "event" + str(i)
        while (frame_num <= end_frame):
            success, frame = videoCapture.read()
            frame_data = log.get(str(frame_num), "None")
            if frame_data != "None":
                obj_data = frame_data.get("obj_data", "None")
                new_obj_data = []
                # common condition
                for j in range(0, len(obj_data)):
                    data_temp = obj_data[j]
                    value_criteria = data_temp["obj_avg_value"] > 0.5
                    sat_criteria = data_temp["obj_avg_sat"] < 0.09
                    if (value_criteria and sat_criteria):
                        new_obj_data.append(data_temp)
                for j in range(0, len(new_obj_data)):
                    data_temp = new_obj_data[j]
                    cv2.rectangle(frame, (int(data_temp["obj_max_y"]), int(data_temp["obj_max_x"])),
                                  (int(data_temp["obj_min_y"]), int(data_temp["obj_min_x"])), (0, 255, 0), 2)
            space = len(str(i))
            cv2.putText(frame, str(i), (360 - (space * 20), 220), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
                        cv2.LINE_AA)
            result_video_writer.write(frame)
            frame_num += 1
