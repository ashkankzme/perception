import random
from dataset_utility import MRFDatasetUtility as mrfdu
from demographic_utils import DemographicUtils
from utils import loadObjectsFromJsonFile


def analyseMRFWorkerStats():
    # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)
    freqWorkersWDemographics = set()
    workerDemographicStats = {
        'age': 0,
        'education': 0,
        'gender': 0,
        'race': 0,
        'mediaConsumptionRegimen': 0
    }
    frequentWorkerDemographicStats = {
        'age': 0,
        'education': 0,
        'gender': 0,
        'race': 0,
        'mediaConsumptionRegimen': 0
    }
    for worker in workers:
        # print(
        #     f'ID: {worker["id"]}, Age: {worker["age"]}, Education: {worker["education"]}, Gender: {worker["gender"]}, '
        #     f'Race: {worker["race"]}, #Annotations: {len(worker["annotatedFrames"])}'
        # )
        # print('------------------')

        if worker['age'] != 'unknown':

            workerDemographicStats['age'] += 1
            if len(worker['annotatedFrames']) >= 100:
                freqWorkersWDemographics.add(worker['id'])
                frequentWorkerDemographicStats['age'] += 1

        if worker['education'] != 'unknown':

            workerDemographicStats['education'] += 1
            if len(worker['annotatedFrames']) >= 100:
                freqWorkersWDemographics.add(worker['id'])
                frequentWorkerDemographicStats['education'] += 1

        if worker['race'] is not None and len(worker['race']) > 0:

            workerDemographicStats['race'] += 1
            if len(worker['annotatedFrames']) >= 100:
                freqWorkersWDemographics.add(worker['id'])
                frequentWorkerDemographicStats['race'] += 1

        if worker['gender'] != 'unknown':

            workerDemographicStats['gender'] += 1
            if len(worker['annotatedFrames']) >= 100:
                freqWorkersWDemographics.add(worker['id'])
                frequentWorkerDemographicStats['gender'] += 1

        if worker['mediaConsumptionRegimen'] is not None and len(worker['mediaConsumptionRegimen']) > 0:

            workerDemographicStats['mediaConsumptionRegimen'] += 1
            if len(worker['annotatedFrames']) >= 100:
                freqWorkersWDemographics.add(worker['id'])
                frequentWorkerDemographicStats['mediaConsumptionRegimen'] += 1

    freqWorkersWithoutDemographics = [worker['id'] for worker in workers if
                                      len(worker['annotatedFrames']) >= 100 and worker[
                                          'id'] not in freqWorkersWDemographics]
    print(workerDemographicStats)
    print('------------------')
    print(frequentWorkerDemographicStats)
    print('------------------')
    print(freqWorkersWDemographics)
    print('------------------')
    print(freqWorkersWithoutDemographics)


def printDemographicWorkerStats():
    testData = loadObjectsFromJsonFile('/local2/ashkank/perception/data/trajectories/leaveoneout/test_trajectories.json')
    # testData = loadObjectsFromJsonFile('../data/trajectories/leaveoneout/test_trajectories.json')
    workerDemographics = {}
    for dp in testData:
        workerId = DemographicUtils.extractWorkerId(dp['X'])
        if workerId not in workerDemographics:
            workerDemographics[workerId] = DemographicUtils.extractHeader(dp['X'])
        else:
            continue

    for workerId, header in workerDemographics.items():
        print(header)
        print(f'test data points: {len([dp for dp in testData if DemographicUtils.extractWorkerId(dp["X"]) == workerId])}')
        print('------------------')


if __name__ == '__main__':
    # analyseMRFWorkerStats()
    printDemographicWorkerStats()