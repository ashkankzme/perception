from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from misinfo_perception_t5 import MisinfoPerceptionT5
from mrf_data_module import MRFDataModule
from mrf_dataset_utility import MRFDatasetUtility as mrfdu


# remove from here
import sys
sys.path.insert(0, '/local2/ashkank/perception/config/')
import default as defaultConfig


if __name__ == '__main__':

    # generates trajectories for training, validation and testing
    # mrfdu.generateTrajectoriesFromMRFDataset()

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
    mrf = MRFDataModule(defaultConfig)
    # mrf.prepare_data()
    model = MisinfoPerceptionT5(1000000)


    trainer = Trainer(accelerator='cuda',
                      strategy='ddp',
                      devices=[1, 2, 3],
                      default_root_dir=defaultConfig.MODEL_PATH + "Checkpoints",
                      callbacks=[early_stop_callback, lr_monitor])
    trainer.fit(model, datamodule=mrf)
    model.model.save_pretrained(defaultConfig.MODEL_PATH + "trained_model")
