def import_para():
    count = 0
    outset_frame = 0
    cycle_point = 0
    channel = 2  # H:0 S:1 V:2
    window_length = 249

    kernel_size1 = (5, 5)
    kernel_size2 = (7, 7)
    threshold_value = 0.07
    median_blur_value = 3
    wait_time = 10
    return count, outset_frame, cycle_point, channel, window_length, kernel_size1, kernel_size2, threshold_value, median_blur_value, wait_time
