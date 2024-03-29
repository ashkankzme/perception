from evaluate_perception_modeling import evaluate, parseEvalConfig
from per_worker_skip_mrf_data_module import PerWorkerSkipMRFDataModule
from utils import loadObjectsFromJsonFile, saveToJsonFile, getFoldIndices
import sys
import numpy as np


def getAvgAndStd(results):
    avg = sum(results) / len(results)
    std = (sum([(r - avg) ** 2 for r in results]) / len(results)) ** 0.5

    return avg, std


if __name__ == '__main__':
    evalConfig = parseEvalConfig(sys.argv)
    evalConfig.BATCH_SIZE *= 8

    print("Loading worker ids...")
    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')
    # evalConfig.MODEL_PATH = '/local2/ashkank/perception/trainedModels/leaveOneOut'
    # evalConfig.DATASET_PATH = '/local2/ashkank/perception/data/trajectories/leaveoneout/'

    # for each worker, evaluate every fold w and w/out demographics.
    # Record avg and std of the results for each worker
    results = []
    for workerId in workerIdsWDemographics:

        print("Evaluating worker: " + workerId)
        workerTrajectoriesFileName = workerId + '_trajectories.json'
        workerTrajectoriesLength = len(loadObjectsFromJsonFile(evalConfig.DATASET_PATH + workerTrajectoriesFileName))

        workerResults = []
        for foldId in range(evalConfig.FOLDS):
            print(workerId + ", fold: " + str(foldId) + ": starting evaluation w demographics...")
            setattr(evalConfig, "MASKED_DEMOGRAPHICS", False)

            testFoldStartIdx, testFoldEndIdx = getFoldIndices(workerTrajectoriesLength, evalConfig.FOLDS, foldId)
            skipIndices = [_ for _ in range(workerTrajectoriesLength) if _ < testFoldStartIdx or _ >= testFoldEndIdx]

            foldDM = PerWorkerSkipMRFDataModule(evalConfig, workerTrajectoriesFileName, workerTrajectoriesLength, skipIndices=skipIndices)
            # workerFoldModelPath = evalConfig.BASE_MODEL_NAME + workerId + '/' # uncomment for zero-shot evaluation
            workerFoldModelPath = evalConfig.MODEL_PATH + '_' + workerId + '_' + str(foldId) + '/'
            acc, f1_misinfo, f1_real = evaluate(evalConfig, foldDM, workerFoldModelPath)
            workerResults.append([acc, f1_misinfo, f1_real])
            print(workerId + ", fold: " + str(foldId) + ": w demographic evaluation results: ", acc, f1_misinfo, f1_real)


            print(workerId + ", fold: " + str(foldId) + ": starting evaluation without demographics...")
            setattr(evalConfig, "MASKED_DEMOGRAPHICS", True)

            foldDM = PerWorkerSkipMRFDataModule(evalConfig, workerTrajectoriesFileName, workerTrajectoriesLength, skipIndices=skipIndices)
            acc, f1_misinfo, f1_real = evaluate(evalConfig, foldDM, workerFoldModelPath)
            workerResults[foldId] += [acc, f1_misinfo, f1_real]
            print(workerId + ", fold: " + str(foldId) + ": without demographic evaluation results: ", acc, f1_misinfo, f1_real)

            print('-' * 25)

        meanWorkerResults = np.mean(workerResults, axis=0)
        stdWorkerResults = np.std(workerResults, axis=0)
        # results w demographics
        cvAcc = format(meanWorkerResults[0]*100, '.2f') + ' ± ' + format(stdWorkerResults[0]*100, '.2f')
        cvF1_misinfo = format(meanWorkerResults[1]*100, '.2f') + ' ± ' + format(stdWorkerResults[1]*100, '.2f')
        cvF1_real = format(meanWorkerResults[2]*100, '.2f') + ' ± ' + format(stdWorkerResults[2]*100, '.2f')
        # results w/out demographics
        cvAcc2 = format(meanWorkerResults[3]*100, '.2f') + ' ± ' + format(stdWorkerResults[3]*100, '.2f')
        cvF1_misinfo2 = format(meanWorkerResults[4]*100, '.2f') + ' ± ' + format(stdWorkerResults[4]*100, '.2f')
        cvF1_real2 = format(meanWorkerResults[5]*100, '.2f') + ' ± ' + format(stdWorkerResults[5]*100, '.2f')

        workerResults.append([cvAcc, cvF1_misinfo, cvF1_real, cvAcc2, cvF1_misinfo2, cvF1_real2])
        results.append(workerResults)

    saveToJsonFile(results, evalConfig.DATASET_PATH + 'test_results_10fold_masked_w_validation.json')
