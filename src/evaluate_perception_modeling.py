from mrf_data_module import MRFDataModule
from misinfo_perception_t5 import MisinfoPerceptionT5
from sklearn.metrics import f1_score, accuracy_score
from pytorch_lightning import Trainer

import sys

from trainingConfigs import default, bigDataSmallModel, bigDataMediumModel, bigDataBigModel, tinyDataSmallModel, \
    smallDataBigBERT, bigDataSmallModelLabelsOnly



if __name__ == '__main__':

    # read first argument from command line, use the input to determine which config file to use
    trainingConfig = None
    configName = sys.argv[1] if len(sys.argv) > 1 else "notSpecified"
    if configName == "default":
        trainingConfig = default
    elif configName == "bigDataSmallModel":
        trainingConfig = bigDataSmallModel
    elif configName == "bigDataMediumModel":
        trainingConfig = bigDataMediumModel
    elif configName == "bigDataBigModel":
        trainingConfig = bigDataBigModel
    elif configName == "tinyDataSmallModel":
        trainingConfig = tinyDataSmallModel
    elif configName == "smallDataBigBERT":
        trainingConfig = smallDataBigBERT
    elif configName == "bigDataSmallModelLabelsOnly":
        trainingConfig = bigDataSmallModelLabelsOnly

    if trainingConfig is None:
        raise Exception("Invalid config name: " + configName)

    trainingConfig.BATCH_SIZE *= 8
    # trainingConfig.MODEL_PATH = '/local2/ashkank/perception/trainedModels/bigDataSmallModelLabelsOnly/trained_model/'
    # trainingConfig.DATASET_PATH = '/local2/ashkank/perception/data/trajectories/biglabelsonly/'

    mrf = MRFDataModule(trainingConfig)
    model = MisinfoPerceptionT5(trainingConfig, len(mrf.test_dataloader()) // trainingConfig.BATCH_SIZE, loadLocally=True, localModelPath=trainingConfig.MODEL_PATH)
    model.testSetJson = mrf.test_dataloader().dataset.jsonData

    trainer = Trainer(accelerator='cuda', devices=[2, 3])
    # trainer = Trainer(accelerator='cuda', devices=[3])
    trainer.test(model, datamodule=mrf)

    yPred, yTrue = model.reduceTestResults()

    print(accuracy_score(yTrue, yPred), f1_score(yTrue, yPred, pos_label=0), f1_score(yTrue, yPred, pos_label=1))