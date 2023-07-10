from mrf_data_module import MRFDataModule
from misinfo_perception_t5 import MisinfoPerceptionT5
from sklearn.metrics import f1_score, accuracy_score
from pytorch_lightning import Trainer

import sys

import trainingConfigs


def evaluate(trainingConfig):
    trainingConfig.BATCH_SIZE *= 8
    # trainingConfig.MODEL_PATH = '/local2/ashkank/perception/trainedModels/bigDataLongMediumModel/'
    # trainingConfig.DATASET_PATH = '/local2/ashkank/perception/data/trajectories/big/'
    mrf = MRFDataModule(trainingConfig)
    model = MisinfoPerceptionT5(trainingConfig, len(mrf.test_dataloader()) // trainingConfig.BATCH_SIZE,
                                loadLocally=True, localModelPath=trainingConfig.MODEL_PATH + 'trained_model/')
    model.testSetJson = mrf.test_dataloader().dataset.jsonData
    trainer = Trainer(accelerator='cuda', devices='auto')
    # trainer = Trainer(accelerator='cuda', devices=[3])
    trainer.test(model, datamodule=mrf)
    yPred, yTrue = model.reduceTestResults()
    print(accuracy_score(yTrue, yPred), f1_score(yTrue, yPred, pos_label=0), f1_score(yTrue, yPred, pos_label=1))


if __name__ == '__main__':

    # read first argument from command line, use the input to determine which config file to use
    trainingConfig = None
    configName = sys.argv[1] if len(sys.argv) > 1 else "notSpecified"
    if configName == "notSpecified":
        raise Exception("config name not specified")

    trainingConfig = getattr(trainingConfigs, configName, None)

    if trainingConfig is None:
        raise Exception("Invalid config name: " + configName)

    evaluate(trainingConfig)