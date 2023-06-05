import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import loadObjectsFromJsonFile
from transformers import T5Tokenizer
from mrf_dataset import MRFDataset


class MRFDataModule(pl.LightningDataModule):
    def __init__(self, datasetConfig):
        super().__init__()
        self.batchSize = datasetConfig.BATCH_SIZE
        self.datasetPath = datasetConfig.DATASET_PATH
        self.maxInputLength = datasetConfig.MAX_INPUT_LENGTH
        self.maxOutputLength = datasetConfig.MAX_OUTPUT_LENGTH
        self.tokenizer = T5Tokenizer.from_pretrained(datasetConfig.BASE_MODEL_NAME)

        self.trainDataLoader = None
        self.valDataLoader = None
        self.testDataLoader = None


    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        # iterate over data and prep it for training (e.g. separate X from y, etc.)
        # self.trainData = loadObjectsFromJsonFile(self.datasetPath + "train_trajectories.json")
        # self.trainData = self.formatData(self.trainData)
        # self.valData = loadObjectsFromJsonFile(self.datasetPath + "eval_trajectories.json")
        # self.valData = self.formatData(self.valData)
        # self.testData = loadObjectsFromJsonFile(self.datasetPath + "test_trajectories.json")
        # self.testData = self.formatData(self.testData)
        pass


    def setup(self, stage='test'):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.

        # if stage == 'train':
        #     self.transformedTrainData = self.transform(self.trainData)
        #
        # elif stage == 'val':
        #     self.transformedValData = self.transform(self.valData)
        #
        # elif stage == 'test':
        #     self.transformedTestData = self.transform(self.testData)
        pass


    def train_dataloader(self):
        if not self.trainDataLoader:
            transformedTrainData = MRFDataset(self.datasetPath + "train_trajectories.json")
            self.trainDataLoader = DataLoader(transformedTrainData, batch_size=self.batchSize, shuffle=False)
        return self.trainDataLoader


    def val_dataloader(self):
        if not self.valDataLoader:
            transformedValData = MRFDataset(self.datasetPath + "eval_trajectories.json")
            self.valDataLoader = DataLoader(transformedValData, batch_size=self.batchSize, shuffle=False)
        return self.valDataLoader


    def test_dataloader(self):
        if not self.testDataLoader:
            transformedTestData = MRFDataset(self.datasetPath + "test_trajectories.json")
            self.testDataLoader = DataLoader(transformedTestData, batch_size=self.batchSize, shuffle=False)
        return self.testDataLoader
