from mrf_dataset import MRFDataset
from demographic_utils import DemographicUtils


class MaskedMRFDataset(MRFDataset):
    def __init__(self, datasetPath, config, removeDemographics=False, excludedWorkers = []):
        super().__init__(datasetPath, config)
        if removeDemographics:
            self.jsonData = self.maskDemographics(self.jsonData)
        if excludedWorkers:
            self.jsonData = self.removeWorkers(self.jsonData, excludedWorkers)
    def maskDemographics(self, data):
        for i, dp in enumerate(data):
            data[i]['X'] = DemographicUtils.maskDemographicHeaderAttributes(dp['X'])
        return data

    def removeWorkers(self, data, excludedWorkers):
        return [dp for dp in data if not len([workerID for workerID in excludedWorkers if workerID in dp['X']])]
