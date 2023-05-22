import torch
import torch.nn as nn
import config.default as defaultConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl


class MisinfoPerceptionT5(pl.LightningModule):
    def __init__(self, dataModule, loadLocally=False, localModelPath=None):
        super().__init__()

        self.dm = dataModule
        self.tokenizer = T5Tokenizer.from_pretrained(defaultConfig.BASE_MODEL_NAME)
        # todo set training config

        if not loadLocally:
            self.model = T5ForConditionalGeneration.from_pretrained(defaultConfig.BASE_MODEL_NAME, device_map="auto")  # check whether the device_map config makes sense
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(localModelPath, device_map="auto")

        # todo check whether freezing the params is necessary
        # freeze all layers after a certain depth, as determined by defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD
        # for param in self.model.parameters()[:defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD]:
        #     param.requires_grad = False

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss


    def forward(self, inputPrompt):
        input_ids = self.tokenizer(inputPrompt, return_tensors="pt").input_ids.to("cuda")  # is this parallelizable?

        outputs = self.model.generate(input_ids)
        print(self.tokenizer.decode(outputs[0]))

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):  # todo fix overriding issue
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * len(self.train_dataloader())
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=self.hparams.warmup_steps,
                                                                     num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def train_dataloader(self):
        if not self.dm.trainData:
            self.dm.setup('train')
        return self.dm.train_dataloader()

    def val_dataloader(self):
        if not self.dm.valData:
            self.dm.setup('val')
        return self.dm.val_dataloader()

    def test_dataloader(self):
        if not self.dm.testData:
            self.dm.setup('test')
        return self.dm.test_dataloader()