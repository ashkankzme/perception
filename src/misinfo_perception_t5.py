import torch
import torch.nn as nn
from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM
import pytorch_lightning as pl


class MisinfoPerceptionT5(pl.LightningModule):
    def __init__(self, config, trainSetLength, loadLocally=False, localModelPath=None, lr=5e-5, num_train_epochs=1, warmup_steps=1000):
        super().__init__()

        self.config = config
        self.trainSetLength = trainSetLength
        # self.tokenizer = T5Tokenizer.from_pretrained(self.config.BASE_MODEL_NAME)
        # todo set training config
        self.hparams.lr = lr
        self.hparams.num_train_epochs = num_train_epochs
        self.hparams.warmup_steps = warmup_steps

        trustRemoteCode = True if self.config.TRUST_REMOTE_CODE else False

        if not loadLocally:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.BASE_MODEL_NAME, trust_remote_code=trustRemoteCode)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(localModelPath, trust_remote_code=trustRemoteCode)

        # todo check whether freezing the params is necessary
        # freeze all layers after a certain depth, as determined by defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD
        # for param in self.model.parameters()[:defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD]:
        #     param.requires_grad = False

    def common_step(self, batch, batch_idx):
        outputs = self(**batch)
        return outputs.loss
        # loss = outputs.loss
        #
        # return loss

    def training_step(self, batch, batch_idx):
        # return self.common_step(batch, batch_idx)
        loss = self.common_step(batch, batch_idx)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)

        return loss


    def forward(self, input_ids, attention_mask, labels=None): # todo test
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # return outputs

    def validation_step(self, batch, batch_idx):
        # return self.common_step(batch, batch_idx)
        loss = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        return self.common_step(batch, batch_idx)
        # loss = self.common_step(batch, batch_idx)
        # return loss

    def configure_optimizers(self):
        # create optimizer
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
        # create learning rate scheduler
        num_train_optimization_steps = self.hparams.num_train_epochs * self.trainSetLength
        lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
                                                                     num_warmup_steps=self.hparams.warmup_steps,
                                                                     num_training_steps=num_train_optimization_steps),
                        'name': 'learning_rate',
                        'interval': 'step',
                        'frequency': 1}

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # def train_dataloader(self):
    #     return self.dm.train_dataloader()
    #
    # def val_dataloader(self):
    #     return self.dm.val_dataloader()
    #
    # def test_dataloader(self):
    #     return self.dm.test_dataloader()


    def evaluateOneExample(self, inputText, expectedOutputText):
        # todo test
        input_ids = self.tokenizer.encode(inputText, return_tensors="pt")
        generated_ids = self.model.generate(input_ids=input_ids, num_beams=4, max_length=5, early_stopping=True)
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        perceivedLabel = self.extractPerceivedLabelFromPrediction(generated_text)
        return expectedOutputText == perceivedLabel


    @staticmethod
    def extractPerceivedLabelFromPrediction(yPred):
        return "" # todo implement