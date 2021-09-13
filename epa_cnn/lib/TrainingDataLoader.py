import os 
import matplotlib.pyplot as plt 
import warnings
warnings.simplefilter('always')
import numpy as np 

from lib.ErrorMessage import CustomError

loading_bar_len = 20
loaded_symbol = "="
unloaded_symbol = "-"

class TrainDataLoader(object):
    '''
    TrainDataLoader(data_root)\n
    use for training.
    
    Method:
        load() # load file from all class.\n
        reshape(target_shape=(N, M, C)) # reshape all class file to target shape.(default target_shape=None)
    '''
    def __init__(self, data_root):
        self.data_root = data_root
        self.class_names = self.__classChecker()

    def __classChecker(self):
        class_name = []
        for pattern in os.listdir(self.data_root):
            class_name.append(pattern)
        return class_name

    def load(self):
        self.raw_data = {}
        for class_ in self.class_names:
            self.raw_data[class_] = []
            data_path = os.path.join(self.data_root, class_)
            class_data_path = os.listdir(data_path)
            data_count = len(class_data_path)
            if(data_count > 0):
                count_ = 0
                for pattern in os.listdir(data_path):
                    count_+=1
                    loaded_len = int(loading_bar_len*count_/data_count)
                    print("\rLoading class: {}, total:{} |{}{}|".format(class_, data_count, loaded_symbol*loaded_len, unloaded_symbol*(loading_bar_len-loaded_len)), end="")
                    temp_data = plt.imread(os.path.join(data_path, pattern))
                    if(self.__channelChecker(temp_data)==-1):
                        temp_data = np.reshape(temp_data, (np.shape(temp_data)[0], np.shape(temp_data)[1], 1))
                    self.raw_data[class_].append(temp_data)
                print("Done.")

            else:
                warnings.warn("Class {} no data, please check.".format(class_))
        
        return self.raw_data

    def __channelChecker(self, data):
        shape = np.shape(data)
        if(len(shape)<3):
            return -1
        elif(len(shape)==3):
            return shape[-1]

    def reshape(self, target_shape=None):
        self.reshaped_data = {} 
        for class_ in self.raw_data.keys():
            count_ = 0
            data_len = len(self.raw_data[class_])
            self.reshaped_data[class_] = []
            if(data_len > 0):
                for data in self.raw_data[class_]:
                    count_+=1
                    processed_len = int(loading_bar_len*count_/data_len)
                    print("\rReshape class: {}, total:{} |{}{}|".format(class_, data_len, loaded_symbol*processed_len, unloaded_symbol*(loading_bar_len-processed_len)), end="")
                    self.reshaped_data[class_].append(np.reshape(data, target_shape))
                print("Done.")
                
        return self.reshaped_data

    def normalization(self, factor=None):
        '''
        factor=None => data/255\n
        factor={"min":0, "max":255} => (data-0)/(255-0)\n
        factor={"mean":0, "std":255} => (data-0)/255
        '''
        def n(fa, fb):
            normalized_data = {} 
            for class_ in self.reshaped_data.keys():
                count_ = 0
                data_len = len(self.reshaped_data[class_])
                normalized_data[class_] = []
                if(data_len > 0):
                    for data in self.reshaped_data[class_]:
                        count_+=1
                        processed_len = int(loading_bar_len*count_/data_len)
                        print("\rNormalization class: {}, total:{} |{}{}|".format(class_, data_len, loaded_symbol*processed_len, unloaded_symbol*(loading_bar_len-processed_len)), end="")
                        normalized_data[class_].append((data-fa)/fb)
                    print("Done.")
                    
            return normalized_data
        if(factor==None):
            self.normalized_data = n(0, 255)
        else:
            min_max = ["min", "max"]
            z_score = ["mean", "std"]
            min_max.sort()
            z_score.sort()
            
            keys = list(factor.keys())
            keys.sort()
            if(keys==min_max):
                self.normalized_data = n(factor[min_max[1]], (factor[min_max[0]]-factor[min_max[1]]))
            elif(keys==z_score):
                self.normalized_data = n(factor[z_score[1]], factor[z_score[0]])
            else:
                raise CustomError("Factor not available.")
        
        return self.normalized_data

