import random
from src.mrf_dataset_utility import MRFDatasetUtility as mrfdu
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)

    testWorkers, trainWorkers, evalWorkers = mrfdu.splitDataset(workers, 0.8, 0.1, 0.1)
    testTrajectories = mrfdu.generateTrajectorySequencesFromMRFDataset(testWorkers, {'min': 3, 'max': 5}, 100, '../data/test_trajectories.txt')
    trainTrajectories = mrfdu.generateTrajectorySequencesFromMRFDataset(trainWorkers, {'min': 3, 'max': 5}, 100, '../data/train_trajectories.txt')
    evalTrajectories = mrfdu.generateTrajectorySequencesFromMRFDataset(evalWorkers, {'min': 3, 'max': 5}, 100, '../data/eval_trajectories.txt')

    # todo initiate decision transformer
    # train, test, val = mrfdu.splitDataset(workers, 0.8, 0.1, 0.1)
    # train architecture, which includes a DT and a PLM
    # evaluate architecture
    # store experiment results