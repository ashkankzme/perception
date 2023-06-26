from mrf_dataset import MRFDataset
from misinfo_perception_t5 import MisinfoPerceptionT5
from transformers import AutoTokenizer

import sys

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

    mrfTestSet = MRFDataset('/local2/ashkank/perception/data/trajectories/big/test_trajectories.json', trainingConfig)
    testSetJson = mrfTestSet.jsonData
    model = MisinfoPerceptionT5(trainingConfig, len(testSetJson) // trainingConfig.BATCH_SIZE, loadLocally=True,
                                localModelPath='/local2/ashkank/perception/trainedModels/bigDataSmallModel/trained_model')

    tokenizer = AutoTokenizer.from_pretrained(trainingConfig.BASE_MODEL_NAME)

    for testInput in testSetJson[:10]:
        print(testInput['X'])
        print('--' * 20)
        input_ids = tokenizer(testInput['X'], return_tensors='pt').input_ids
        output_ids = model.model.generate(input_ids)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f'predicted: {output}')
        print(f'actual: {testInput["y"]}')
