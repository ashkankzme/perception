import random
from utils import MRFDatasetUtility as mrfdu
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')

    # randomSample = random.sample(refinedGroupedMTurkData[random.choice(list(refinedGroupedMTurkData.keys()))], 20)
    # for sample in randomSample:
    #     print(sample)
    #     print('------------------')
    #
