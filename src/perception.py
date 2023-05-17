import random
from src.mrf_dataset_utility import MRFDatasetUtility as mrfdu
from src.trajectory import Trajectory
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)
    workers = workers[:60]

    trainEvalCutOffIndex = int(len(workers) * 0.8)
    evalTestCutOffIndex = int(len(workers) * 0.9)
    trainWorkers, evalWorkers, testWorkers = workers[:trainEvalCutOffIndex], workers[trainEvalCutOffIndex:evalTestCutOffIndex], workers[evalTestCutOffIndex:]
    # trainTrajectories = mrfdu.generateTrajectorySequencesFromMRFDataset(trainWorkers, {'min': 3, 'max': 5}, 100, '../data/train_trajectories.txt')
    evalTrajectories = Trajectory.generateTrajectorySequencesFromMRFDataset(evalWorkers, {'min': 4, 'max': 6}, 100, '../data/eval_trajectories.txt')
    # testTrajectories = mrfdu.generateTrajectorySequencesFromMRFDataset(testWorkers, {'min': 3, 'max': 5}, 100, '../data/test_trajectories.txt')

    # todo initiate decision transformer
    # train, test, val = mrfdu.splitDataset(workers, 0.8, 0.1, 0.1)
    # train architecture, which includes a DT and a PLM
    # evaluate architecture
    # store experiment results