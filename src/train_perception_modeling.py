from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from misinfo_perception_t5 import MisinfoPerceptionT5
from mrf_data_module import MRFDataModule
from mrf_dataset_utility import MRFDatasetUtility as mrfdu
import time, sys

import trainingConfigs


if __name__ == '__main__':

    # read first argument from command line, use the input to determine which config file to use
    trainingConfig = None
    configName = sys.argv[1] if len(sys.argv) > 1 else "notSpecified"
    if configName == "notSpecified":
        raise Exception("config name not specified")

    trainingConfig = getattr(trainingConfigs, configName, None)

    if trainingConfig is None:
        raise Exception("Invalid config name: " + configName)

    dataGeneration = False
    dataGenFlag = sys.argv[2] if len(sys.argv) > 2 else "notSpecified"
    if dataGenFlag == "gen":
        dataGeneration = True
    elif dataGenFlag == "load":
        dataGeneration = False
    else:
        print("Invalid or empty data generation flag: " + dataGenFlag)

    if dataGeneration:
        labelsOnly = trainingConfig.LABELS_ONLY if hasattr(trainingConfig, "LABELS_ONLY") else False
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
    mrf = MRFDataModule(trainingConfig)
    # mrf.prepare_data()
    model = MisinfoPerceptionT5(trainingConfig, len(mrf.train_dataloader())//trainingConfig.BATCH_SIZE)

    accumulatedBatches = trainingConfig.ACCUMULATED_BATCHES if hasattr(trainingConfig, "ACCUMULATED_BATCHES") else 1

    trainer = Trainer(accelerator='cuda',
                      strategy='ddp',
                      devices='auto',
                      default_root_dir=trainingConfig.MODEL_PATH + "Checkpoints",
                      callbacks=[lr_monitor],
                      # callbacks=[early_stop_callback, lr_monitor],
                      accumulate_grad_batches=accumulatedBatches,
                      precision=16,)
    trainer.fit(model, datamodule=mrf)
    model.model.save_pretrained(trainingConfig.MODEL_PATH + "trained_model")
