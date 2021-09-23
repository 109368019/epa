import json
import numpy as np
import os

from para.parameter import Parameter
import csv

date = "0705_genv_A17"
log_path = "./log/" + date + "/"
event_path = "./event/" + date + "/"
if not os.path.isdir(event_path):
    os.mkdir(event_path)
log_list = [i for i in os.listdir(log_path) if (i[-4::] == "json")]
# 排序 清單 
log_list = sorted(log_list)
# print(log_list)

for log_name in log_list:

    with open(log_path + log_name, 'r') as fp:
        log = json.load(fp)

    # print(int(log.keys()))
    if len(log.keys()) > 0:
        last_num = int(list(log.keys())[-1])
        first_num = int(list(log.keys())[0])

    print(first_num)
    print(last_num)
    # initial setting
    #  
    stop_event_buffer = 2
    single_test = 0
    non_event_count = 0
    # event_num = -1

    event_real = []  # [[start_frame,end_frame],[]...]
    event_temp = []  # [[start_frame_num, non_event_count, [extra_condition]],...] 當 non_event_count >= 2 pop to event_real
    condition = 1

    for frame_num in range(first_num, last_num):
        # nun = input()
        # if nun =="q":
        #     break
        frame_data = log.get(str(frame_num), "None")
        # print(frame_data)
        print(frame_num)
        # print(event_real)
        # print(event_temp)
        if frame_data != "None":
            obj_data = frame_data.get("obj_data", "None")
            new_obj_data = []
            # common condition
            for i in range(0, len(obj_data)):
                # print(obj_data)
                data_temp = obj_data[i]
                value_criteria = data_temp["obj_avg_value"] > 0.5
                sat_criteria = data_temp["obj_avg_sat"] < 0.09
                if (value_criteria and sat_criteria and data_temp["obj_brighten"]):
                    new_obj_data.append(data_temp)
            # if ((len(new_obj_data) == 0) and (len(event_temp) == 0)):
            # print(new_obj_data[])
            # else:
            # compare event_temp and obj
            for i in range(0, len(event_temp)):
                condition_list = event_temp[i][2]
                event_flag = 0
                for j in range(0, len(new_obj_data)):
                    obj_data_dict = new_obj_data[i]
                    #
                    #
                    #
                    if (condition) == 1:
                        event_flag = 1
                        event_temp[i][1] = 0
                        # updata event criteria
                        del new_obj_data[j]
                        break
                if event_flag == 0:
                    # non event count ++
                    event_temp[i][1] += 1
            # if new obj_event add event_temp
            if len(new_obj_data) >= 1:
                for i in range(0, len(new_obj_data)):
                    # event condition => event_temp
                    save_event = new_obj_data[i]
                    condition_save = 1
                    event_temp.append([frame_num, 0, condition_save])
        # merge similar event
        else:
            if len(event_temp) >= 1:
                for i in range(0, len(event_temp)):
                    event_temp[i][1] += 1

        merge_event_temp = []
        for i in range(0, len(event_temp)):
            condition_buffer = event_temp[i]
            if condition_buffer == -1:
                continue
            condition_i = condition_buffer[2]
            for j in range(i + 1, len(event_temp)):

                condition_j = event_temp[j][2]
                if condition_i == condition_j:

                    sfn = min(event_temp[i][0], event_temp[j][0])
                    nec = min(event_temp[i][1], event_temp[j][1])
                    if event_temp[i][0] < event_temp[j][0]:
                        condition_buffer = [sfn, nec, condition_j]
                    else:
                        condition_buffer = [sfn, nec, condition_i]
                    event_temp[j] = -1
            merge_event_temp.append(condition_buffer)
        # check non event count             
        event_temp = []
        for i in range(0, len(merge_event_temp)):
            if merge_event_temp[i][1] < 2:
                event_temp.append(merge_event_temp[i])
            else:
                event_real.append([merge_event_temp[i][0], frame_num])

    print(event_real)
    print(event_temp)
    event_name = event_path + log_name[0:-5]
    with open(event_name + ".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in event_real:
            writer.writerow(i)
