
class Parameter(object):

    def __init__(self):
        self.count = 0
        self.outset_frame = 0
        self.cycle_point = 0
        self.channel = 2  # H:0 S:1 V:2
        self.window_length = 35
        
        self.kernel_size1 = (5, 5)
        self.kernel_size2 = (7, 7)
        self.threshold_value = 0.07
        self.median_blur_value = 3
        self.wait_time = 10
        # input_video = './綠環境-(A15)家電冷氣機設備投入口及處理線、冷媒吸取區-20200331114723-20200306-150000-movie.mp4'
        # return count, outset_frame, cycle_point, channel, window_length, kernel_size1, kernel_size2, threshold_value, median_blur_value, wait_time, input_video
