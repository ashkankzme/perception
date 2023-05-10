import random
from utils import MRFDatasetUtility as mrfdu
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)

    workerDemographicStats = {
        'age': 0,
        'education': 0,
        'gender': 0,
        'race': 0
    }

    frequentWorkerDemographicStats = {
        'age': 0,
        'education': 0,
        'gender': 0,
        'race': 0
    }

    for worker in workers:
        # print(
        #     f'ID: {worker["id"]}, Age: {worker["age"]}, Education: {worker["education"]}, Gender: {worker["gender"]}, '
        #     f'Race: {worker["race"]}, #Annotations: {len(worker["annotatedFrames"])}'
        # )
        # print('------------------')

        if worker['age'] is not None:
            workerDemographicStats['age'] += 1
            if len(worker['annotatedFrames']) > 100:
                frequentWorkerDemographicStats['age'] += 1
        if worker['education'] is not None:
            workerDemographicStats['education'] += 1
            if len(worker['annotatedFrames']) > 100:
                frequentWorkerDemographicStats['education'] += 1
        if worker['race'] is not None:
            workerDemographicStats['race'] += 1
            if len(worker['annotatedFrames']) > 100:
                frequentWorkerDemographicStats['race'] += 1
        if worker['gender'] is not None:
            workerDemographicStats['gender'] += 1
            if len(worker['annotatedFrames']) > 100:
                frequentWorkerDemographicStats['gender'] += 1

    print(workerDemographicStats)
    print('------------------')
    print(frequentWorkerDemographicStats)
