import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import loadObjectsFromJsonFile
from transformers import AutoTokenizer
from mrf_dataset import MRFDataset
from masked_mrf_dataset import MaskedMRFDataset


class MRFDataModule(pl.LightningDataModule):
    def __init__(self, datasetConfig, excludedWorkers = []):
        super().__init__()
        self.config = datasetConfig
        self.batchSize = datasetConfig.BATCH_SIZE
        self.datasetPath = datasetConfig.DATASET_PATH
        self.maxInputLength = datasetConfig.MAX_INPUT_LENGTH
        self.maxOutputLength = datasetConfig.MAX_OUTPUT_LENGTH
        trustRemoteCode = True if datasetConfig.TRUST_REMOTE_CODE else False
        self.tokenizer = AutoTokenizer.from_pretrained(datasetConfig.BASE_MODEL_NAME, trust_remote_code=trustRemoteCode)
        self.maskedDemographics = getattr(datasetConfig, 'MASKED_DEMOGRAPHICS', False)
        self.excludedWorkers = excludedWorkers

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

        # if not self.trainDataLoader:
        #     self.trainDataLoader = self._wrapInDatasetObj("train_trajectories.json")
        # if not self.valDataLoader:
        #     self.valDataLoader = self._wrapInDatasetObj("eval_trajectories.json")
        # if not self.testDataLoader:
        #     self.testDataLoader = self._wrapInDatasetObj("test_trajectories.json")
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
            self.trainDataLoader = self._wrapInDatasetObj("train_trajectories.json")
        return self.trainDataLoader


    def val_dataloader(self):
        if not self.valDataLoader:
            self.valDataLoader = self._wrapInDatasetObj("eval_trajectories.json")
        return self.valDataLoader


    def test_dataloader(self):
        if not self.testDataLoader:
            self.testDataLoader = self._wrapInDatasetObj("test_trajectories.json")
        return self.testDataLoader


    def _wrapInDatasetObj(self, fileName):
        transformedData = MaskedMRFDataset(self.datasetPath + fileName, self.config,
                                           removeDemographics=self.maskedDemographics,
                                           excludedWorkers=self.excludedWorkers)
        # transformedData.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        return DataLoader(transformedData, batch_size=self.batchSize, shuffle=False, num_workers=1, pin_memory=True)
