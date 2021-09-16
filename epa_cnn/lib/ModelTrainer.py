# basic lib
import numpy as np
np.random.seed(1210)
import os
# import sys
# import pickle
import random
random.seed(1210)
from sklearn.model_selection import train_test_split

# custom lib 
from lib.TrainingDataLoader import TrainDataLoader # training data preprocessor
# from lib.Preprocessor import Preprocessor # single preprocessor

# tensorflow lib
import tensorflow as tf 
tf.random.set_seed(1210)
from tensorflow import keras as K
from tensorflow.keras import layers as L
# import keras_tuner as kt

# solve cudnn error and disable message
from lib import solveCudnnError 
solveCudnnError.solve_cudnn_error()
solveCudnnError.diable_tensorflow_warning()

class ModelTrainer(object):
    def __init__(self, model_version:str, data_size:tuple=(28, 28)):
        self.data_size = data_size

        self.model_version = "{}.h5".format(model_version)
        self.default_model_path = "models_structure"
        self.__dirChecker(self.default_model_path)

    def __modelDefinition(self): # define your model
        input_layer = L.Input(shape=(*self.data_size,3))

        hiden_layer = L.Conv2D(filters=256, kernel_size=7, activation="relu", padding="same")(input_layer)
        hiden_layer = L.Conv2D(filters=256, kernel_size=5, activation="relu", padding="same")(hiden_layer)
        hiden_layer = L.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(hiden_layer)
        hiden_layer = L.BatchNormalization()(hiden_layer)
        high_way = L.Conv2D(filters=256, kernel_size=3, activation="relu", padding="same")(input_layer)
        hiden_layer = L.Concatenate()([hiden_layer, high_way])
        hiden_layer = L.MaxPooling2D(pool_size=(2, 2))(hiden_layer)

        hiden_layer = L.Conv2D(filters=256, kernel_size=3, activation="relu")(hiden_layer)
        hiden_layer = L.MaxPooling2D(pool_size=(2, 2))(hiden_layer)
        hiden_layer = L.BatchNormalization()(hiden_layer)

        hiden_layer = L.Conv2D(filters=512, kernel_size=3, activation="relu")(hiden_layer)
        hiden_layer = L.MaxPooling2D(pool_size=(2, 2))(hiden_layer)
        hiden_layer = L.BatchNormalization()(hiden_layer)

        hiden_layer = L.Conv2D(filters=32, kernel_size=1, activation="relu")(hiden_layer)

        hiden_layer = L.Flatten()(hiden_layer)
        output_layer = L.Dense(units=2, activation="sigmoid")(hiden_layer)

        model = K.Model(inputs=input_layer, outputs=output_layer)
        return model

    def __dirChecker(self, path):
        print("Checking file path ({})...".format(path), end="")
        if(not os.path.isdir(path)):
            print("Fail.")
            print("Build file path ({})".format(path))
            os.mkdir(path)
            return False
        else:
            print("Exist.")
            return True
    
    def __fileChecker(self, path):
        print("Checking file ({})...".format(path), end="")
        if(not os.path.isfile(path)):
            print("Fail.")
            return False
        else:
            print("Exist.")
            return True

    def build(self, summary:bool=True):
        model_path = os.path.join(self.default_model_path, self.model_version)
        if(self.__fileChecker(model_path)):
            print("Load model.")
            self.model = K.models.load_model(model_path)
        else:
            print("Build model.")
            self.model = self.__modelDefinition()
            self.model.save(model_path, save_format="h5")

        if(summary):
            self.model.summary()

        return self.model

    def loadTrainingData(self, root_path:str, factor:dict=None, class_list:list=None):
        tdl = TrainDataLoader(root_path)
        tdl.class_names = class_list
        tdl.load()
        tdl.resize(self.data_size)
        tdl.normalization(factor=factor)
        self.x_train = []
        self.y_train = []

        self.x_test = []
        self.y_test = []

        count_ = 0
        for class_ in tdl.normalized_data.keys():
            if(len(tdl.normalized_data[class_])!=0):
                class_data = tdl.normalized_data[class_]
                class_label = list(np.ones((len(tdl.normalized_data[class_]),), dtype=int)*count_)
                
                X, x, Y, y = train_test_split(class_data, class_label, test_size=0.25)
                
                self.x_train = self.x_train + list(X)
                self.x_test = self.x_test + list(x)
                
                self.y_train = self.y_train + list(Y)
                self.y_test = self.y_test + list(y)
                
                count_+=1

        self.x_train = np.array(self.x_train)
        self.x_test = np.array(self.x_test)
        self.y_train = np.eye(count_)[self.y_train]
        self.y_test = np.eye(count_)[self.y_test]

        return self.x_train, self.x_test, self.y_train, self.y_test
    
    def save(self):
        model_path = os.path.join(self.default_model_path, self.model_version+"_trained")
        self.model.save(model_path, save_format="h5")

    # TODO reload training status
    # def __parameterLoader(self):
    #     if(os.path.isfile(self.parameter_path)):
    #         with open(self.parameter_path, "rb") as handle:
    #             self.status = pickle.load(handle)
    #         return True
    #     else:
    #         return False            

    # def rework(self):
    #     iswork = self.__parameterLoader()
    #     if(not iswork):
    #         parameters = {}
        
    
