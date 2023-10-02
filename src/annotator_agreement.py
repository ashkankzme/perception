from utils import loadObjectsFromJsonFile, saveToJsonFile
from misinfo_perception_t5 import MisinfoPerceptionT5
from collections import Counter
import krippendorff


def generateSentimentAnalysisDataForUsersWDemographics():

    # worker IDs for different groups:
    menIds = ["A002160837SWJFPIAI7L7", "AGQKBH53S12SZ", "ADJ9I7ZBFYFH7", "A3LRZX8477TYYZ", "A2CWJRAEFZ44HU",
              "A17Q4QN6UE0EZC", "A5TWD5QD99GZY"]
    womenIds = ["AWNR2FOWI73GL", "A2OFN0A5CPLH57", "A324VBRLXHG5IB", "A2OVX9UW5WANQE", "A3UJX60MALFMW0", "A32W24TWSWXW",
                "A173A97OFDAX9F", "A3MD34XEB4H6JF", "A2C84POENS2UNY", "A2RMJNF6IPI42F"]
    whiteIds = ["A2RMJNF6IPI42F", "A2C84POENS2UNY", "ADJ9I7ZBFYFH7", "A3LRZX8477TYYZ", "A32W24TWSWXW", "A3UJX60MALFMW0",
                "A2OVX9UW5WANQE", "A324VBRLXHG5IB", "A2OFN0A5CPLH57", "A5TWD5QD99GZY", "AWNR2FOWI73GL"]
    nonWhiteIds = ["AGQKBH53S12SZ", "A2CWJRAEFZ44HU", "A3MD34XEB4H6JF", "A173A97OFDAX9F", "A002160837SWJFPIAI7L7",
                   "A17Q4QN6UE0EZC"]
    collegeEducatedIds = ["AGQKBH53S12SZ", "A2C84POENS2UNY", "AWNR2FOWI73GL", "A2RMJNF6IPI42F", "A17Q4QN6UE0EZC",
                          "A32W24TWSWXW", "A002160837SWJFPIAI7L7", "A173A97OFDAX9F", "A2OFN0A5CPLH57", "A3LRZX8477TYYZ"]
    nonCollegeEducatedIds = ["A324VBRLXHG5IB", "ADJ9I7ZBFYFH7", "A5TWD5QD99GZY", "A2CWJRAEFZ44HU", "A3UJX60MALFMW0",
                             "A2OVX9UW5WANQE", "A3MD34XEB4H6JF"]
    youngIds = ["ADJ9I7ZBFYFH7", "A3UJX60MALFMW0", "A32W24TWSWXW", "A002160837SWJFPIAI7L7", "A173A97OFDAX9F"]
    oldIds = ["A17Q4QN6UE0EZC", "A2OFN0A5CPLH57", "A3LRZX8477TYYZ", "A2OVX9UW5WANQE", "A3JC9VPPTHNKVL",
              "A2RMJNF6IPI42F", "A3MD34XEB4H6JF", "AWNR2FOWI73GL", "A2CWJRAEFZ44HU", "A5TWD5QD99GZY", "A2C84POENS2UNY",
              "AGQKBH53S12SZ", "A324VBRLXHG5IB"]

    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = [worker for worker in workers if len(worker['annotatedFrames']) >= 100]  # throwing out workers with less than 10 annotated frames
    demographicWorkers = [worker for worker in workers if worker['id'] in workerIdsWDemographics]

    headlinesMappedToReactions = {}
    for worker in demographicWorkers:
        for frame in worker['annotatedFrames']:
            if len(frame['reaction']) == 0:
                continue

            headline = frame['Input.sentence']
            if headline not in headlinesMappedToReactions:
                headlinesMappedToReactions[headline] = []
            perceivedLabel = MisinfoPerceptionT5._extractLabel(frame['perceivedLabel'])
            headlinesMappedToReactions[headline].append({
                'workerId': worker['id'],
                'reaction': perceivedLabel
            })

    # keep only the headlines that have at least 2 reactions
    headlinesMappedToReactions = {headline: reactions for headline, reactions in headlinesMappedToReactions.items() if len(reactions) >= 2}
    # calculate aa, ab, bb matrices for each group
    aa, ab, bb = calculateValueCounts(headlinesMappedToReactions, menIds, womenIds)
    # calculate Krippendorf's alpha using these three matrices
    mmA = krippendorff.alpha(value_counts=aa, level_of_measurement='nominal')
    mfA = krippendorff.alpha(value_counts=ab, level_of_measurement='nominal')
    ffA = krippendorff.alpha(value_counts=bb, level_of_measurement='nominal')
    print('male-male, male-female, female-female')
    print(mmA, mfA, ffA)
    print('*' * 20)

    aa, ab, bb = calculateValueCounts(headlinesMappedToReactions, whiteIds, nonWhiteIds)
    # calculate Krippendorf's alpha using these three matrices
    wwA = krippendorff.alpha(value_counts=aa, level_of_measurement='nominal')
    wnA = krippendorff.alpha(value_counts=ab, level_of_measurement='nominal')
    nnA = krippendorff.alpha(value_counts=bb, level_of_measurement='nominal')
    print('white-white, white-nonwhite, nonwhite-nonwhite')
    print(wwA, wnA, nnA)
    print('*' * 20)

    aa, ab, bb = calculateValueCounts(headlinesMappedToReactions, collegeEducatedIds, nonCollegeEducatedIds)
    # calculate Krippendorf's alpha using these three matrices
    ccA = krippendorff.alpha(value_counts=aa, level_of_measurement='nominal')
    cnA = krippendorff.alpha(value_counts=ab, level_of_measurement='nominal')
    nnA = krippendorff.alpha(value_counts=bb, level_of_measurement='nominal')
    print('college-college, college-noncollege, noncollege-noncollege')
    print(ccA, cnA, nnA)
    print('*' * 20)

    aa, ab, bb = calculateValueCounts(headlinesMappedToReactions, youngIds, oldIds)
    # calculate Krippendorf's alpha using these three matrices
    yyA = krippendorff.alpha(value_counts=aa, level_of_measurement='nominal')
    yoA = krippendorff.alpha(value_counts=ab, level_of_measurement='nominal')
    ooA = krippendorff.alpha(value_counts=bb, level_of_measurement='nominal')
    print('young-young, young-old, old-old')
    print(yyA, yoA, ooA)
    print('*' * 20)


def calculateValueCounts(headlinesMappedToReactions, groupA, groupB):
    aa, ab, bb = [], [], []
    for headline, reactions in headlinesMappedToReactions.items():
        headlineAnnotators = [reaction['workerId'] for reaction in reactions]
        groupAHeadlineAnnotators = [annotator for annotator in headlineAnnotators if annotator in groupA]
        groupBHeadlineAnnotators = [annotator for annotator in headlineAnnotators if annotator in groupB]
        if len(groupAHeadlineAnnotators) > 0 and len(groupBHeadlineAnnotators) > 0:
            dist = Counter([reaction['reaction'] for reaction in reactions])
            ab.append([dist[0], dist[1]])
        elif len(groupAHeadlineAnnotators) > 0:
            dist = Counter([reaction['reaction'] for reaction in reactions if reaction['workerId'] in groupA])
            aa.append([dist[0], dist[1]])
        elif len(groupBHeadlineAnnotators) > 0:
            dist = Counter([reaction['reaction'] for reaction in reactions if reaction['workerId'] in groupB])
            bb.append([dist[0], dist[1]])

    return aa, ab, bb


if __name__ == '__main__':
    generateSentimentAnalysisDataForUsersWDemographics()