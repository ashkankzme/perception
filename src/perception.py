import random
from utils import MRFDatasetUtility as mrfdu
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    mTurkData = mrfdu.readCSVFiles('../data/mturk_data/')
    print("Number of HITs completed: ", len(mTurkData))
    groupedMTurkData = mrfdu.groupByWorkerId(mTurkData)
    refinedGroupedMTurkData = mrfdu.filterColumns(groupedMTurkData)
    workers = mrfdu.transformMTurkData(mTurkData, refinedGroupedMTurkData)

    # persisting processed data
    saveObjectsToJsonFile(workers, '../data/mrf_turk_processed.json') # todo test serialization/deserialization

    randomSample = random.sample(refinedGroupedMTurkData[random.choice(list(refinedGroupedMTurkData.keys()))], 20)
    for sample in randomSample:
        print(sample)
        print('------------------')

