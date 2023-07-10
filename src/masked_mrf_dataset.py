from mrf_dataset import MRFDataset
from demographic_utils import DemographicUtils


class MaskedMRFDataset(MRFDataset):
    def __init__(self, datasetPath, config):
        super().__init__(datasetPath, config)
        self.jsonData = self.maskDemographics(self.jsonData)

    def maskDemographics(self, data):
        for i, dp in enumerate(data):
            data[i]['X'] = DemographicUtils.maskDemographicHeaderAttributes(dp['X'])
        return data
