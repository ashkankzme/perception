from mrf_dataset import MRFDataset
from demographic_utils import DemographicUtils


class MaskedMRFDataset(MRFDataset):
    def __init__(self, datasetPath, config, removeDemographics=False, excludedWorkers=None, skipIndices=None):
        super().__init__(datasetPath, config)
        if skipIndices is None:
            skipIndices = []
        if excludedWorkers is None:
            excludedWorkers = []
        if removeDemographics:
            self.jsonData = self.maskDemographics(self.jsonData)
        if excludedWorkers:
            self.jsonData = self.removeWorkers(self.jsonData, excludedWorkers)
        if skipIndices:
            self.jsonData = self.skip(self.jsonData, skipIndices)
    def maskDemographics(self, data):
        for i, dp in enumerate(data):
            data[i]['X'] = DemographicUtils.maskDemographicHeaderAttributes(dp['X'])
        return data

    def removeWorkers(self, data, excludedWorkers):
        return [dp for dp in data if not len([workerID for workerID in excludedWorkers if workerID in dp['X']])]

    def skip(self, data, skipIndices):
        return [dp for i, dp in enumerate(data) if i not in skipIndices]
