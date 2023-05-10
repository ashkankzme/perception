import random
from utils import MRFDatasetUtility as mrfdu
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)

    workersWDemographics = 0
    for worker in workers:
        print(
            f'ID: {worker["id"]}, Age: {worker["age"]}, Education: {worker["education"]}, Gender: {worker["gender"]}, '
            f'Race: {worker["race"]}, #Annotations: {len(worker["annotatedFrames"])}'
        )
        print('------------------')

    print(len([a for a in workers if len(a['annotatedFrames']) > 100]))