# basic lib
import numpy as np
from tensorflow.python.autograph.operators.py_builtins import all_
np.random.seed(1210)
# import os
# import sys
import pickle
import matplotlib.pyplot as plt

# custom lib 
from lib.ModelTrainer import ModelTrainer
from lib.TrainingDataLoader import TrainDataLoader # training data preprocessor
# from lib.Preprocessor import Preprocessor # single preprocessor

# tensorflow lib
import tensorflow as tf 
# from tensorflow import keras as K
# from tensorflow.keras import layers as L
# import keras_tuner as kt

# solve cudnn error and disable message
from lib import solveCudnnError 
solveCudnnError.solve_cudnn_error()
solveCudnnError.diable_tensorflow_warning()

if __name__ == "__main__":
    factor = {"min":0, "max":255}
    mt = ModelTrainer("ResNet_2_class_red", (64,64))
    mt.loadTrainingData(root_path="train_data", factor=factor, class_list=["neg_red", "ref_red"])
    print(mt.x_train.mean(), mt.x_train.std(), mt.x_train.min(), mt.x_train.max())
    mt.build()

    model = mt.model
    model.compile("adam", "categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(mt.x_train, mt.y_train, batch_size=8, epochs=30, validation_data=(mt.x_test, mt.y_test))
    results = model.evaluate(mt.x_test, mt.y_test, batch_size=32)
    print("test loss: {}, test acc: {}".format(results[0], results[1]))

    plt.subplot(2,1,1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title("loss history")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(history.history["accuracy"], label="train_accuracy")
    plt.plot(history.history["val_accuracy"], label="val_accuracy")
    plt.title("accuracy history")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_curve.png")

    mt.save()
