# basic lib
import numpy as np
import os, sys

#custom lib 
from lib.TrainingDataLoader import TrainDataLoader # training data preprocessor
from lib.Preprocessor import Preprocessor # single preprocessor

# tensorflow lib
import tensorflow as tf 
from tensorflow import keras as K
from tensorflow.keras import layers as L
import keras_tuner as kt

# solve cudnn error and disable message
from lib import solveCudnnError 
solveCudnnError.solve_cudnn_error()
solveCudnnError.diable_tensorflow_warning()

class ModelTrainer(object):
    def __init__(self):
        pass
