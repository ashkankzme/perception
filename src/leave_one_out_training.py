from utils import loadObjectsFromJsonFile
from train_perception_modeling import parseArgs, train
from mrf_data_module import MRFDataModule
from dataset_utility import MRFDatasetUtility as mrfdu
import sys, time


if __name__ == '__main__':
    trainingConfig, dataGeneration = parseArgs(sys.argv)

    print("Loading worker ids...")
    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')
    workerIdsWithoutDemographics = loadObjectsFromJsonFile('../data/worker_ids_without_demographics.json')

    if dataGeneration:
        labelsOnly = getattr(trainingConfig, "LABELS_ONLY", False)
        # generates trajectories for training, validation and testing
        mrfdu.generateLeaveOneOutTrajectories(trainingConfig.DATASET_PATH, labelsOnly=labelsOnly, testWorkerIds=workerIdsWDemographics) # uncomment for trajectory data generation
        # mrfdu.generateLeaveOneOutQueries(trainingConfig.DATASET_PATH, labelsOnly=labelsOnly, testWorkerIds=workerIdsWDemographics) # uncomment for single frame data generation
        # wait for i/o to finish
        time.sleep(10)

    setattr(trainingConfig, 'SAVE_MODEL', True)

    # train a base model with all the non-demographic workers, save results
    print("Training base model...")
    mrf = MRFDataModule(trainingConfig, excludedWorkers=workerIdsWDemographics)
    train(trainingConfig, mrf, trainingConfig.MODEL_PATH+'_base/trained_model/', lossMonitor='training_loss')
    print("Base model trained.")

    # leave one out training
    for workerId in workerIdsWDemographics:
        # load model locally, exclude non-demographic workers + current worker from training
        print("Training model with worker " + workerId + " excluded...")
        mrf = MRFDataModule(trainingConfig, excludedWorkers=[workerId]+workerIdsWithoutDemographics)
        train(trainingConfig, mrf, trainingConfig.MODEL_PATH+'_loo_'+workerId+'/', loadLocally=True,
              localModelPath=trainingConfig.MODEL_PATH+'_base/trained_model/', lossMonitor='training_loss') # todo refactor here, add eval data
        print("Model trained with worker " + workerId + " excluded.")
