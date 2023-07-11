from utils import loadObjectsFromJsonFile
from train_perception_modeling import parseArgs
import sys


if __name__ == '__main__':
    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')
    trainingConfig, dataGeneration = parseArgs(sys.argv)