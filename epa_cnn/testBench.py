from lib.TrainingDataLoader import TrainDataLoader 

if __name__ == "__main__":
    dl = TrainDataLoader("train_data")
    dl.load()
    dl.reshape()
    dl.normalization()