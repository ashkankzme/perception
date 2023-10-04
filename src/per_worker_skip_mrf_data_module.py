from mrf_data_module import MRFDataModule

class PerWorkerSkipMRFDataModule(MRFDataModule):
    def __init__(self, datasetConfig, datasetFileName, workerTrajectoriesLength, excludedWorkers=None, skipIndices=None, validationIndices=None):
        super().__init__(datasetConfig, excludedWorkers, skipIndices)
        self.datasetFileName = datasetFileName
        self.validationIndices = validationIndices
        self.workerTrajectoriesLength = workerTrajectoriesLength

    def train_dataloader(self):
        if not self.trainDataLoader:
            self.trainDataLoader = self._wrapInDatasetObj(self.datasetFileName, additionalSkipIndices=self.validationIndices)
        return self.trainDataLoader

    def test_dataloader(self):
        if not self.testDataLoader:
            self.testDataLoader = self._wrapInDatasetObj(self.datasetFileName)
        return self.testDataLoader

    def val_dataloader(self):
        skipIndices = [_ for _ in range(self.workerTrajectoriesLength) if _ not in self.validationIndices]
        if not self.valDataLoader:
            self.valDataLoader = self._wrapInDatasetObj(self.datasetFileName, additionalSkipIndices=skipIndices)
        return self.valDataLoader