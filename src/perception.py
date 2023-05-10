import random
from src.mrf_dataset_utility import MRFDatasetUtility as mrfdu
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)
    # todo initiate decision transformer
    # train, test, val = mrfdu.splitDataset(workers, 0.8, 0.1, 0.1)
    # train architecture, which includes a DT and a PLM
    # evaluate architecture
    # store experiment results