from mrf_data_module import MRFDataModule

class PerWorkerSkipMRFDataModule(MRFDataModule):
    def __init__(self, datasetConfig, datasetFileName, excludedWorkers=None, skipIndices=None):
        super().__init__(datasetConfig, excludedWorkers, skipIndices)
        self.datasetFileName = datasetFileName

    def train_dataloader(self):
        if not self.trainDataLoader:
            self.trainDataLoader = self._wrapInDatasetObj(self.datasetFileName)
        return self.trainDataLoader

    def test_dataloader(self):
        if not self.testDataLoader:
            self.testDataLoader = self._wrapInDatasetObj(self.datasetFileName)
        return self.testDataLoader