import numpy as np
import cv2 as cv2 

from lib.ErrorMessage import CustomError

class Preprocessor(object):
    '''
    use for single data.
    '''
    def __init__(self, shape, factor=None):
        self.shape = shape 
        self.factor = factor

    def resize(self, data):
        return cv2.resize(data, self.shape, interpolation=cv2.INTER_CUBIC)
    
    def normalization(self, data):
        if(self.factor==None):
            return data/255
        else:
            min_max = ["min", "max"]
            z_score = ["mean", "std"]
            min_max.sort()
            z_score.sort()
            
            keys = list(self.factor.keys())
            keys.sort()
            if(keys==min_max):
                return (data-self.factor[min_max[1]])/(self.factor[min_max[0]]-self.factor[min_max[1]])
            elif(keys==z_score):
                return (data-self.factor[z_score[1]])/self.factor[z_score[0]]
            else:
                raise CustomError("Factor not available.")