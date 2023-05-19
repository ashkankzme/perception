import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import loadObjectsFromJsonFile


class MRFDataModule(pl.LightningDataModule):
    def __init__(self, datasetConfig):
        super().__init__()
        self.batchSize = datasetConfig.BATCH_SIZE
        self.datasetPath = datasetConfig.DATASET_PATH
        self.trainData = None
        self.valData = None
        self.testData = None


    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        # todo iterate over data and prep it for training (e.g. separate X from y, etc.)
        pass


    def setup(self, stage='test'):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.

        if stage == 'train':
            self.trainData = loadObjectsFromJsonFile(self.datasetPath + "train_trajectories.json")

        elif stage == 'val':
            self.valData = loadObjectsFromJsonFile(self.datasetPath + "eval_trajectories.json")

        elif stage == 'test':
            self.testData = loadObjectsFromJsonFile(self.datasetPath + "test_trajectories.json")


    def train_dataloader(self):
        # todo set random seed so shuffle becomes predictable
        return DataLoader(self.trainData, batch_size=self.batchSize, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.valData, batch_size=self.batchSize, shuffle=False)
        # Return DataLoader for Validation Data here


    def test_dataloader(self):
        return DataLoader(self.testData, batch_size=self.batchSize, shuffle=False)
        # Return DataLoader for Testing Data here