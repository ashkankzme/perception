from utils import loadObjectsFromJsonFile, getFoldIndices
from train_perception_modeling import parseArgs, train
from per_worker_skip_mrf_data_module import PerWorkerSkipMRFDataModule
from dataset_utility import MRFDatasetUtility as mrfdu
import sys, time


if __name__ == '__main__':
    trainingConfig, dataGeneration = parseArgs(sys.argv)

    print("Loading worker ids...")
    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')

    if dataGeneration:
        labelsOnly = getattr(trainingConfig, "LABELS_ONLY", False)
        # generates trajectories for training, validation and testing
        mrfdu.generateTrajectoriesPerWorker(trainingConfig.DATASET_PATH, labelsOnly=labelsOnly, testWorkerIds=workerIdsWDemographics)
        print("Trajectories generated, waiting for io to finish...")
        # wait for i/o to finish
        time.sleep(10)


    setattr(trainingConfig, 'SAVE_MODEL', True)

    # for each worker, train N different models for cross validation
    for workerId in workerIdsWDemographics:

        print("Starting training models for worker " + workerId)
        workerTrajectoriesFileName = workerId + '_trajectories.json'
        workerTrajectoriesLength = len(loadObjectsFromJsonFile(trainingConfig.DATASET_PATH + workerTrajectoriesFileName))
        localModelPath = trainingConfig.BASE_MODEL_NAME + workerId + '/'

        for foldId in range(trainingConfig.FOLDS):
            print(workerId + ", fold: " + str(foldId) + ": starting training...")
            testFoldStartIdx, testFoldEndIdx = getFoldIndices(workerTrajectoriesLength, trainingConfig.FOLDS, foldId)
            validationFoldId = foldId - 1 if foldId > 0 else trainingConfig.FOLDS - 1
            validationFoldStartIdx, validationFoldEndIdx = getFoldIndices(workerTrajectoriesLength, trainingConfig.FOLDS, validationFoldId)

            foldDM = PerWorkerSkipMRFDataModule(trainingConfig, workerTrajectoriesFileName, workerTrajectoriesLength,
                                                skipIndices=[_ for _ in range(testFoldStartIdx, testFoldEndIdx)],
                                                validationIndices=[_ for _ in range(validationFoldStartIdx, validationFoldEndIdx)])

            modelOutputPath = trainingConfig.MODEL_PATH + '_' + workerId + '_' + str(foldId) + '/'

            train(trainingConfig, foldDM, modelOutputPath, loadLocally=True, localModelPath=localModelPath, lossMonitor='validation_loss')
            print(workerId + ", fold: " + str(foldId) + ": trained")
