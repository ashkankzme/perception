import random
from utils import MRFDatasetUtility as mrfdu


if __name__ == '__main__':
    mTurkData = mrfdu.readCSVFiles('../data/mturk_data/')
    print("Number of HITs completed: ", len(mTurkData))
    groupedMTurkData = mrfdu.groupByWorkerId(mTurkData)
    refinedGroupedMTurkData = mrfdu.filterColumns(groupedMTurkData)
    print(random.sample(refinedGroupedMTurkData[random.choice(list(refinedGroupedMTurkData.keys()))], 1))

