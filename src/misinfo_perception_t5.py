from transformers import AdamW, get_linear_schedule_with_warmup, AutoModelForSeq2SeqLM, AutoTokenizer
import pytorch_lightning as pl
from utils import loadObjectsFromJsonFile


class MisinfoPerceptionT5(pl.LightningModule):
    def __init__(self, config, trainSetLength, loadLocally=False, localModelPath=None, lr=5e-5, num_train_epochs=1, warmup_steps=1000):
        super().__init__()

        self.config = config
        self.trainSetLength = trainSetLength
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.BASE_MODEL_NAME)

        self.hparams.lr = lr
        self.hparams.num_train_epochs = num_train_epochs
        self.hparams.warmup_steps = warmup_steps

        trustRemoteCode = True if self.config.TRUST_REMOTE_CODE else False

        if not loadLocally:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.BASE_MODEL_NAME, trust_remote_code=trustRemoteCode)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(localModelPath, trust_remote_code=trustRemoteCode)

        self.testResults = []
        self.testSetJson = None

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

    def reduceTestResults(self):
        yPred, yTrue = [], []
        for result in self.testResults:
            yPred.extend(result['yPred'])
            yTrue.extend(result['yTrue'])

        return yPred, yTrue

    def test_step(self, batch, batch_idx):
        # return self.common_step(batch, batch_idx)
        # loss = self.common_step(batch, batch_idx)
        # return loss
        # if self.testSetJson is None:
        #     self.testSetJson = loadObjectsFromJsonFile(self.config.DATASET_PATH + 'test_trajectories.json')

        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = [dp['y'] for dp in self.testSetJson[batch_idx * self.config.BATCH_SIZE: (batch_idx + 1) * self.config.BATCH_SIZE]]
        outputs_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=self.config.MAX_OUTPUT_LENGTH, num_beams=4)

        yPred = []
        yTrue = []

        for i, output_ids in enumerate(outputs_ids):
            output = self.tokenizer.decode(output_ids, max_length=self.config.MAX_OUTPUT_LENGTH, padding="max_length", truncation=True)
            gTruth = labels[i]
            predictedLabel = MisinfoPerceptionT5._extractLabel(output)
            yPred.append(predictedLabel)
            actualLabel = MisinfoPerceptionT5._extractLabel(gTruth)
            yTrue.append(actualLabel)

        for i, y in enumerate(yPred):
            if y == -1:
                yPred[i] = 1 - yTrue[i]  # swaping out the label with the wrong answer, since the output format was corrupt (penalizing in evaluation)

        self.testResults.append({'yPred': yPred, 'yTrue': yTrue})


    @staticmethod
    def _extractLabel(text):
        return 0 if 'Perceived Label: misinformation' in text else 1 if 'Perceived Label: real news' in text else -1
        # return 'misinformation' if 'Perceived Label: misinformation' in text else 'real news' if 'Perceived Label: real news' in text else 'unknown'


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
