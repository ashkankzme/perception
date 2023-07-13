from utils import loadObjectsFromJsonFile
from train_perception_modeling import parseArgs, train
from mrf_data_module import MRFDataModule
import sys, time


if __name__ == '__main__':
    trainingConfig, dataGeneration = parseArgs(sys.argv)



    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')
    workerIdsWithoutDemographics = loadObjectsFromJsonFile('../data/worker_ids_without_demographics.json')

    # train a base model with all the non-demographic workers, save results
    mrf = MRFDataModule(trainingConfig, excludedWorkers=workerIdsWDemographics)
    train(trainingConfig, mrf, trainingConfig.MODEL_OUTPUT_PATH+'_base/')

    # todo properly tear down the model and data in memory to avoid OOM errors

    # leave one out training
    for workerId in workerIdsWDemographics:
        # load model locally, exclude non-demographic workers + current worker from training
        mrf = MRFDataModule(trainingConfig, excludedWorkers=[workerId]+workerIdsWithoutDemographics)
        train(trainingConfig, mrf, trainingConfig.MODEL_OUTPUT_PATH+'_loo_'+workerId+'/', loadLocally=True,
              localModelPath=trainingConfig.MODEL_OUTPUT_PATH+'_base/trained_model/')

        # todo properly tear down the model and data in memory to avoid OOM errors

    # leave one out evaluation
    for workerId in workerIdsWDemographics:
        # load model trained explicitly for this worker, evaluate on this worker only
        print()
        # todo
        # todo properly tear down the model and data in memory to avoid OOM errors