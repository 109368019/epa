# basic lib
import numpy as np
np.random.seed(1210)
# import os
# import sys
# import pickle

# custom lib 
from lib.ModelTrainer import ModelTrainer
# from lib.TrainingDataLoader import TrainDataLoader # training data preprocessor
# from lib.Preprocessor import Preprocessor # single preprocessor

# tensorflow lib
import tensorflow as tf 
# from tensorflow import keras as K
# from tensorflow.keras import layers as L
import keras_tuner as kt

# solve cudnn error and disable message
from lib import solveCudnnError 
solveCudnnError.solve_cudnn_error()
solveCudnnError.diable_tensorflow_warning()

if __name__ == "__main__":
    mt = ModelTrainer("test_version", (28,28))
    mt.loadTrainingData("train_data")
    mt.build()

    model = mt.model
    model.compile("adam", "mse")
    model.fit(mt.x_train, mt.y_train, batch_size=8, epochs=2)

