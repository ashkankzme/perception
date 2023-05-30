import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl


# remove from here
import sys
sys.path.insert(0, '/local2/ashkank/perception/config/')
import default as defaultConfig



class MisinfoPerceptionT5(pl.LightningModule):
    def __init__(self, dataModule, loadLocally=False, localModelPath=None, lr=5e-5, warmup_steps=1000):
        super().__init__()

        self.dm = dataModule
        self.tokenizer = T5Tokenizer.from_pretrained(defaultConfig.BASE_MODEL_NAME)
        # todo set training config
        self.hparams.lr = lr
        self.hparams.warmup_steps = warmup_steps

        if not loadLocally:
            self.model = T5ForConditionalGeneration.from_pretrained(defaultConfig.BASE_MODEL_NAME, device_map="auto")  # check whether the device_map config makes sense
        else:
            self.model = T5ForConditionalGeneration.from_pretrained(localModelPath, device_map="auto")

        # todo check whether freezing the params is necessary
        # freeze all layers after a certain depth, as determined by defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD
        # for param in self.model.parameters()[:defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD]:
        #     param.requires_grad = False

        self.save_hyperparameters()
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


    def forward(self, input_ids, attention_mask, labels=None): # todo test
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs

    def validation_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)

        return loss

    def configure_optimizers(self):
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
            self.dm.prepare_data()
            self.dm.setup('train')
        return self.dm.train_dataloader()

    def val_dataloader(self):
        if not self.dm.valData:
            self.dm.prepare_data()
            self.dm.setup('val')
        return self.dm.val_dataloader()

    def test_dataloader(self):
        if not self.dm.testData:
            self.dm.prepare_data()
            self.dm.setup('test')
        return self.dm.test_dataloader()


    def evaluateOneExample(self, inputText, expectedOutputText):
        # todo test
        input_ids = self.tokenizer.encode(inputText, return_tensors="pt")
        generated_ids = self.model.generate(input_ids=input_ids, num_beams=4, max_length=5, early_stopping=True)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        perceivedLabel = self.extractPerceivedLabelFromPrediction(generated_text)
        return expectedOutputText == perceivedLabel


    @staticmethod
    def extractPerceivedLabelFromPrediction(text):
        return "" # todo implement