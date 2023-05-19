from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from misinfo_perception_t5 import MisinfoPerceptionT5
from mrf_data_module import MRFDataModule

if __name__ == '__main__':
    # todo initiate decision transformer
    # train, test, val = mrfdu.splitDataset(workers, 0.8, 0.1, 0.1)
    # train architecture, which includes a DT and a PLM
    # evaluate architecture
    # store experiment results


    # for early stopping, see https://pytorch-lightning.readthedocs.io/en/1.0.0/early_stopping.html?highlight=early%20stopping
    early_stop_callback = EarlyStopping(
        monitor='validation_loss',
        patience=3,
        strict=False,
        verbose=False,
        mode='min'
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    mrfTrain = MRFDataModule()
    model = MisinfoPerceptionT5(mrfTrain)  # todo set training config

    trainer = Trainer(#gpus=1,
                      default_root_dir="../models/1_first_run/Checkpoints",
                      callbacks=[early_stop_callback, lr_monitor])
    trainer.fit(model)
