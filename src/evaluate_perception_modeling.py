from mrf_data_module import MRFDataModule
from misinfo_perception_t5 import MisinfoPerceptionT5
from sklearn.metrics import f1_score, accuracy_score
from pytorch_lightning import Trainer

import sys

import trainingConfigs


def evaluate(trainingConfig, mrf, localModelPath):
    trainingConfig.BATCH_SIZE *= 8
    model = MisinfoPerceptionT5(trainingConfig, len(mrf.test_dataloader()) // trainingConfig.BATCH_SIZE,
                                loadLocally=True, localModelPath=localModelPath)
    model.testSetJson = mrf.test_dataloader().dataset.jsonData
    trainer = Trainer(accelerator='cuda', devices=1)
    # trainer = Trainer(accelerator='cuda', devices=[3])
    trainer.test(model, datamodule=mrf)
    yPred, yTrue = model.reduceTestResults()
    return accuracy_score(yTrue, yPred), f1_score(yTrue, yPred, pos_label=0), f1_score(yTrue, yPred, pos_label=1)



def parseEvalConfig(args):
    configName = args[1] if len(args) > 1 else "notSpecified"
    if configName == "notSpecified":
        raise Exception("config name not specified")
    trainingConfig = getattr(trainingConfigs, configName, None)
    if trainingConfig is None:
        raise Exception("Invalid config name: " + configName)

    return trainingConfig


if __name__ == '__main__':

    trainingConfig = parseEvalConfig(sys.argv)
    # trainingConfig.MODEL_PATH = '/local2/ashkank/perception/trainedModels/bigDataSmallModelLabelsOnly/'
    # trainingConfig.DATASET_PATH = '/local2/ashkank/perception/data/trajectories/biglabelsonly/'
    mrf = MRFDataModule(trainingConfig)
    print(trainingConfig.DESCRIPTION)
    acc, f1_misinfo, f1_real = evaluate(trainingConfig, mrf, trainingConfig.MODEL_PATH+'trained_model')
    print(acc, f1_misinfo, f1_real)
    print('-' * 30)