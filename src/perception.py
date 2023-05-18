import random
from src.mrf_dataset_utility import MRFDatasetUtility as mrfdu
from src.trajectory import Trajectory
from utils import saveObjectsToJsonFile, loadObjectsFromJsonFile


if __name__ == '__main__':
    # workers = mrfdu.loadAndCleanMRFDataset('../data/mturk_data/', '../data/mrf_turk_processed.json')
    workers = loadObjectsFromJsonFile('../data/mrf_turk_processed.json')
    # workers = sorted(workers, key=lambda x: len(x['annotatedFrames']), reverse=True)
    workers = [worker for worker in workers if len(worker['annotatedFrames']) >= 20] # throwing out workers with less than 10 annotated frames
    random.seed(1372)
    random.shuffle(workers)

    trainEvalCutOffIndex = int(len(workers) * 0.8)
    evalTestCutOffIndex = int(len(workers) * 0.9)
    trainWorkers, evalWorkers, testWorkers = workers[:trainEvalCutOffIndex], workers[trainEvalCutOffIndex:evalTestCutOffIndex], workers[evalTestCutOffIndex:]
    trainTrajectories = Trajectory.generateTrajectorySequencesFromMRFDataset(trainWorkers, {'min': 4, 'max': 8}, 1000000, '../data/trajectories/1_initial/train_trajectories.json')
    evalTrajectories = Trajectory.generateTrajectorySequencesFromMRFDataset(evalWorkers, {'min': 4, 'max': 8}, 1000000, '../data/trajectories/1_initial/eval_trajectories.json')
    testTrajectories = Trajectory.generateTrajectorySequencesFromMRFDataset(testWorkers, {'min': 4, 'max': 8}, 1000000, '../data/trajectories/1_initial/test_trajectories.json')

    # todo initiate decision transformer
    # train, test, val = mrfdu.splitDataset(workers, 0.8, 0.1, 0.1)
    # train architecture, which includes a DT and a PLM
    # evaluate architecture
    # store experiment results