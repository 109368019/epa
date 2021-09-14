# basic lib
import os, sys

# tensorflow lib
import tensorflow as tf 
from tensorflow import keras as K
from tensorflow.keras import layers as L

# solve cudnn error and disable message
from lib import solveCudnnError 
solveCudnnError.solve_cudnn_error()
solveCudnnError.diable_tensorflow_warning()

class ModelBuilder(object):
    def __init__(self, model_version):
        self.default_model_path = "models"
        self.__dirChecker(self.default_model_path)

        self.model_version = model_version

    def __modelDefinition(self): # define your model
        input_layer = L.Input(shape=(28,28,3))
        hiden_layer = L.Conv2D(filters=256, kernel_size=3, activation="relu")(input_layer)
        hiden_layer = L.Conv2D(filters=256, kernel_size=3, activation="relu")(hiden_layer)
        hiden_layer = L.Conv2D(filters=512, kernel_size=3, activation="relu")(hiden_layer)
        hiden_layer = L.Conv2D(filters=512, kernel_size=3, activation="relu")(hiden_layer)
        hiden_layer = L.Flatten()(hiden_layer)
        output_layer = L.Dense(units=3, activation="sigmoid")(hiden_layer)
        model = K.Model(inputs=input_layer, outputs=output_layer)
        return model

    def __dirChecker(self, path):
        print("Checking file path ({})...".format(path), end="")
        if(not os.isdir(path)):
            print("Fail.")
            print("Build file path ({})".format(path))
            os.mkdir(path)
            return False
        else:
            print("Exist.")
            return True
    
    def __fileChecker(self, path):
        print("Checking file ({})...".format(path), end="")
        if(not os.isfile(path)):
            print("Fail.")
            return False
        else:
            print("Exist.")
            return True

    def build(self, summary=True):
        if(self.__fileChecker(self.model_version)):
            self.model = K.models.load_model(os.path.join(self.default_model_path, self.model_version))
        else:
            self.model = self.__modelDefinition()

        if(summary):
            self.model.summary()
        return self.model

        