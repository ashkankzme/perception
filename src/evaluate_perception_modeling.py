from mrf_data_module import MRFDataModule
from misinfo_perception_t5 import MisinfoPerceptionT5
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score
from torch.cuda import is_available

import sys, random

from trainingConfigs import default, bigDataSmallModel, bigDataMediumModel, bigDataBigModel, tinyDataSmallModel, \
    smallDataBigBERT



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

    if trainingConfig is None:
        raise Exception("Invalid config name: " + configName)

    trainingConfig.BATCH_SIZE *= 25
    trainingConfig.MODEL_PATH = '/local2/ashkank/perception/trainedModels/bigDataSmallModel/trained_model/'
    trainingConfig.DATASET_PATH = '/local2/ashkank/perception/data/trajectories/big/'

    mrf = MRFDataModule(trainingConfig)
    model = MisinfoPerceptionT5(trainingConfig, len(mrf.test_dataloader()) // trainingConfig.BATCH_SIZE, loadLocally=True, localModelPath=trainingConfig.MODEL_PATH)
    mrfTestSetJson = mrf.test_dataloader().dataset.jsonData

    device = 'cuda:3' if is_available() else 'cpu'
    model.model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(trainingConfig.BASE_MODEL_NAME)
