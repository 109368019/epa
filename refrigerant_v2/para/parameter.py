class Parameter(object):
    # 綠環境-(B5)家電冷氣機冷媒吸取區、冷氣機及洗衣機處理線-20210305-080000
    def __init__(self):
        self.start_frame = 0 #1000
        self.window_length = 1000  # 總統計數量
        self.median_blur_value = 3
        self.probability_throuhold = 10
        self.eps_value = 1  # 30
        self.min_samples_size = 1  # 70
        self.ratio_limit = 0.3
        self.class_map_num_value = 600
        self.mean_value_limit = 0.5
        self.mean_satuation_limit = 0.09 #0.07

        # self.gsussian_sigmaX = 3
        self.gaussian_kernal_size = (3, 3)
        self.predict_class = 1

