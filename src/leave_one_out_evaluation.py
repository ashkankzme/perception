from evaluate_perception_modeling import evaluate, parseEvalConfig
from mrf_data_module import MRFDataModule
from utils import loadObjectsFromJsonFile
import sys


if __name__ == '__main__':
    evalConfig = parseEvalConfig(sys.argv)

    print("Loading worker ids...")
    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')

    # trainingConfig.MODEL_PATH = '/local2/ashkank/perception/trainedModels/bigDataSmallModelLabelsOnly/'
    # trainingConfig.DATASET_PATH = '/local2/ashkank/perception/data/trajectories/biglabelsonly/'
    mrf = MRFDataModule(evalConfig)
    print(evalConfig.DESCRIPTION)
    acc, f1_misinfo, f1_real = evaluate(evalConfig, mrf, evalConfig.MODEL_PATH + '_base/trained_model')
    print(acc, f1_misinfo, f1_real)
    print('-' * 30)