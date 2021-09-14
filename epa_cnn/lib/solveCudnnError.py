import os
import tensorflow as tf 

def diable_tensorflow_warning(LEVEL=3):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "{}".format(LEVEL)

def solve_cudnn_error():
    tf.keras.backend.set_floatx('float64')
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def multiGPU():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Create 2 virtual GPUs with 1GB memory each
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.set_memory_growth(gpus, True),
                tf.config.experimental.set_memory_growth(gpus, True)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
            
# solve_cudnn_error()
# multiGPU()