import numpy as np 
import warnings
warnings.simplefilter('always')

from lib.ErrorMessage import CustomError

class Preprocessor(object):
    '''
    use for single data.
    '''
    def __init__(self, shape, factor=None):
        self.shape = shape 
        self.factor = factor

    def reshape(self, data):
        return np.reshape(data, self.shape)
    
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