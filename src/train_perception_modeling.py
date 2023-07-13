import gc

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from misinfo_perception_t5 import MisinfoPerceptionT5
from mrf_data_module import MRFDataModule
from mrf_dataset_utility import MRFDatasetUtility as mrfdu
import time, sys

import trainingConfigs


def train(trainingConfig, dataGeneration, mrf, modelOutputPath, loadLocally=False, localModelPath=None):
    if dataGeneration:
        labelsOnly = getattr(trainingConfig, "LABELS_ONLY", False)
        # generates trajectories for training, validation and testing
        mrfdu.generateTrajectoriesFromMRFDataset(trainingConfig.DATASET_PATH, labelsOnly=labelsOnly)
        # wait for i/o to finish
        time.sleep(60)
    # https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/T5/Fine_tune_CodeT5_for_generating_docstrings_from_Ruby_code.ipynb#scrollTo=gq2-xLG_alXH
    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    model = MisinfoPerceptionT5(trainingConfig, len(mrf.train_dataloader()) // trainingConfig.BATCH_SIZE,
                                loadLocally=loadLocally, localModelPath=localModelPath)
    accumulatedBatches = trainingConfig.ACCUMULATED_BATCHES if hasattr(trainingConfig, "ACCUMULATED_BATCHES") else 1
    trainer = Trainer(accelerator='cuda',
                      strategy='ddp',
                      devices='auto',
                      default_root_dir=modelOutputPath + "Checkpoints",
                      callbacks=[early_stop_callback, lr_monitor],
                      accumulate_grad_batches=accumulatedBatches, )
    # precision=16,)
    trainer.fit(model, datamodule=mrf)
    model.model.save_pretrained(modelOutputPath + "trained_model")
    model = None
    trainer = None
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    time.sleep(5)


def parseArgs(args):
    configName = args[1] if len(args) > 1 else "notSpecified"
    if configName == "notSpecified":
        raise Exception("config name not specified")
    trainingConfig = getattr(trainingConfigs, configName, None)
    if trainingConfig is None:
        raise Exception("Invalid config name: " + configName)
    dataGeneration = False
    dataGenFlag = args[2] if len(args) > 2 else "notSpecified"
    if dataGenFlag == "gen":
        dataGeneration = True
    elif dataGenFlag == "load":
        dataGeneration = False
    else:
        print("Invalid or empty data generation flag: " + dataGenFlag)

    return trainingConfig, dataGeneration


if __name__ == '__main__':

    trainingConfig, dataGeneration = parseArgs(sys.argv)
    mrf = MRFDataModule(trainingConfig)
    train(trainingConfig, dataGeneration, mrf, trainingConfig.MODEL_PATH)
