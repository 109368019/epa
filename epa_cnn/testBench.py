from lib.TrainingDataLoader import TrainDataLoader 
from lib.ModelTrainer import ModelTrainer
if __name__ == "__main__":
    dl = TrainDataLoader("train_data")
    dl.load()
    dl.resize()
    dl.normalization()

    mt = ModelTrainer()