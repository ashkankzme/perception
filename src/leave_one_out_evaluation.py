from evaluate_perception_modeling import evaluate, parseEvalConfig
from mrf_data_module import MRFDataModule
from utils import loadObjectsFromJsonFile, saveObjectsToJsonFile
import sys

if __name__ == '__main__':
    evalConfig = parseEvalConfig(sys.argv)

    print("Loading worker ids...")
    workerIdsWDemographics = loadObjectsFromJsonFile('../data/worker_ids_with_demographics.json')
    results = []
    evalConfig.MODEL_PATH = '/local2/ashkank/perception/trainedModels/leaveOneOut'
    evalConfig.DATASET_PATH = '/local2/ashkank/perception/data/trajectories/leaveoneout/'

    # evaluate the base model with all the demographic workers
    print("Evaluating the base model...")
    mrf = MRFDataModule(evalConfig)
    acc, f1_misinfo, f1_real = evaluate(evalConfig, mrf, evalConfig.MODEL_PATH + '_base/trained_model')
    results.append({'base': [acc, f1_misinfo, f1_real]})
    print("Base model evaluation results: ", acc, f1_misinfo, f1_real)

    print("Evaluating the base model with masked demographics...")
    setattr(evalConfig, "MASKED_DEMOGRAPHICS", True)
    mrf = MRFDataModule(evalConfig)
    acc, f1_misinfo, f1_real = evaluate(evalConfig, mrf, evalConfig.MODEL_PATH + '_base/trained_model')
    results[-1]['base'] += [acc, f1_misinfo, f1_real]
    print("Base model evaluation results with masked demographics: ", acc, f1_misinfo, f1_real)
    setattr(evalConfig, "MASKED_DEMOGRAPHICS", False)
    print('-' * 25)

    for workerId in workerIdsWDemographics:
        print("Evaluating the leave one out model for worker " + workerId + "...")
        mrf = MRFDataModule(evalConfig, excludedWorkers=[wId for wId in workerIdsWDemographics if wId != workerId])
        acc, f1_misinfo, f1_real = evaluate(evalConfig, mrf, evalConfig.MODEL_PATH + '_loo_' + workerId)
        results.append({workerId: [acc, f1_misinfo, f1_real]})
        print("Worker " + workerId + " model evaluation results: ", acc, f1_misinfo, f1_real)

        print("Evaluating the leave one out model for worker " + workerId + " with masked demographics...")
        setattr(evalConfig, "MASKED_DEMOGRAPHICS", True)
        mrf = MRFDataModule(evalConfig, excludedWorkers=[wId for wId in workerIdsWDemographics if wId != workerId])
        acc, f1_misinfo, f1_real = evaluate(evalConfig, mrf, evalConfig.MODEL_PATH + '_loo_' + workerId)
        results[-1][workerId] += [acc, f1_misinfo, f1_real]
        print("Worker " + workerId + " model evaluation results with masked demographics: ", acc, f1_misinfo, f1_real)
        setattr(evalConfig, "MASKED_DEMOGRAPHICS", False)
        print('-' * 25)

    saveObjectsToJsonFile(results, evalConfig.DATASET_PATH + 'test_results.json')
