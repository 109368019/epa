import csv
import json
import os

date = "0625"
log_path = "../result/20210705/綠環境/B2/log/"
# log_path = "./log/" + date + "/"
event_path = "../result/20210705/綠環境/B2/event/"
# event_path = "./event/" + date + "/"
if not os.path.isdir(event_path):
    os.mkdir(event_path)
log_list = [i for i in os.listdir(log_path) if (i[-4::] == "json")]
log_list = sorted(log_list)
for log_name in log_list:
    with open(log_path + log_name, 'r') as fp:
        log = json.load(fp)

    if len(log.keys()) > 0:
        last_num = int(list(log.keys())[-1])
        first_num = int(list(log.keys())[0])

    print(first_num)
    print(last_num)
    # initial setting
    #
    MINI_LENGTH = 2
    stop_event_buffer = 7
    # single_test = 0
    non_event_count = 0
    # event_num = -1

    event_real = []  # [[start_frame,end_frame],[]...]
    event_temp = []  # [[start_frame_num, non_event_count, [extra_condition]],...] 當 non_event_count >= 2 pop to event_real
    condition = 1

    for frame_num in range(first_num, last_num):
        frame_data = log.get(str(frame_num), "None")
        print(frame_num)
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

            for i in range(0, len(event_temp)):
                condition_list = event_temp[i][2]
                event_flag = False
                for j in range(0, len(new_obj_data)):
                    obj_data_dict = new_obj_data[i]
                    #
                    #
                    #
                    if (condition) == 1:  # 判斷某物件是否屬於某事件
                        event_flag = True
                        event_temp[i][1] = 0
                        # updata event criteria
                        del new_obj_data[j]
                        break
                if not event_flag:
                    # non event count ++
                    event_temp[i][1] += 1
            # if new obj_event add event_temp
            if len(new_obj_data) >= 1:
                for i in range(0, len(new_obj_data)):
                    # event condition => event_temp
                    save_event = new_obj_data[i]
                    condition_save = 1
                    event_temp.append([frame_num, 0, condition_save])  # 產生新事件
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
                    start_frame_num = min(event_temp[i][0], event_temp[j][0])
                    nec = min(event_temp[i][1], event_temp[j][1])
                    if event_temp[i][0] < event_temp[j][0]:
                        condition_buffer = [start_frame_num, nec, condition_j]
                    else:
                        condition_buffer = [start_frame_num, nec, condition_i]
                    event_temp[j] = -1
            merge_event_temp.append(condition_buffer)
        # check non event count             
        event_temp = []
        for i in range(0, len(merge_event_temp)):
            if merge_event_temp[i][1] < stop_event_buffer:
                event_temp.append(merge_event_temp[i])
            else:
                event_len = frame_num - merge_event_temp[i][0] + 1 - stop_event_buffer
                if event_len > MINI_LENGTH:
                    event_real.append([merge_event_temp[i][0], frame_num - stop_event_buffer])

    # plt hist===================================
    # plt.figure(figsize=(14, 6))
    # event_len_sta_list = []
    # for each_event in event_real:
    #     event_len = each_event[1] - each_event[0] + 1 - stop_event_buffer
    #     event_len_sta_list.append(event_len)
    # if len(event_len_sta_list) > 0:
    #     max_len = max(event_len_sta_list)
    #     print("times", len(event_len_sta_list), "max len", max_len)
    # print("log_name", log_name)
    # plt.hist(event_len_sta_list, bins=50, range=(0, 50), color='lightblue', label="Count")
    # plt.legend()
    # plt.xticks([i for i in range(0, 51)])
    # plt.xlabel('event_length(frame)')
    # plt.ylabel('Count(time)')
    # plt.show()

    event_name = event_path + log_name[0:-5]
    with open(event_name + ".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in event_real:
            writer.writerow(i)

if __name__ == "__main__":
    print("123")
    pass