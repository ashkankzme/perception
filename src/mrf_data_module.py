import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils import loadObjectsFromJsonFile
from transformers import T5Tokenizer
from datasets import Dataset


class MRFDataModule(pl.LightningDataModule):
    def __init__(self, datasetConfig):
        super().__init__()
        self.batchSize = datasetConfig.BATCH_SIZE
        self.datasetPath = datasetConfig.DATASET_PATH
        self.maxInputLength = datasetConfig.MAX_INPUT_LENGTH
        self.maxOutputLength = datasetConfig.MAX_OUTPUT_LENGTH
        self.tokenizer = T5Tokenizer.from_pretrained(datasetConfig.BASE_MODEL_NAME)
        self.trainData = None
        self.valData = None
        self.testData = None


    def prepare_data(self):
        # Define steps that should be done
        # on only one GPU, like getting data.
        # iterate over data and prep it for training (e.g. separate X from y, etc.)
        self.trainData = loadObjectsFromJsonFile(self.datasetPath + "train_trajectories.json")
        self.trainData = self.formatData(self.trainData)
        self.valData = loadObjectsFromJsonFile(self.datasetPath + "eval_trajectories.json")
        self.valData = self.formatData(self.valData)
        self.testData = loadObjectsFromJsonFile(self.datasetPath + "test_trajectories.json")
        self.testData = self.formatData(self.testData)


    def formatData(self, data):
        newData = []
        for i, trajectory in enumerate(data):
            dataPoint = {'X': '', 'y': ''}
            formattedTrajectory = '<header> ' + trajectory['header'] + ' </header>'
            for j, frame in enumerate(trajectory['inputFrames']):
                formattedTrajectory += f' <frame_{j}> ' + frame + ' </frame_{j}>'

            formattedTrajectory += ' <query> ' + trajectory['query'] + ' </query>'
            dataPoint['X'] = formattedTrajectory
            dataPoint['y'] = trajectory['prediction']
            newData.append(dataPoint)

        return newData


    def transform(self, data):

        inputs = [dp['X'] for dp in data]
        outputs = [dp['y'] for dp in data]

        model_inputs = self.tokenizer(inputs, max_length=self.maxInputLength, padding="max_length", truncation=True)

        # encode the summaries
        labels = self.tokenizer(outputs, max_length=self.maxOutputLength, padding="max_length", truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        # model_inputs["labels"] = labels_with_ignore_index
        model_inputs = {
            "input_ids": model_inputs.input_ids,
            "token_type_ids": model_inputs.token_type_ids,
            "attention_mask": model_inputs.attention_mask,
            "labels": labels_with_ignore_index
        }

        return Dataset.from_dict(model_inputs)


    def setup(self, stage='test'):
        # Define steps that should be done on
        # every GPU, like splitting data, applying
        # transform etc.

        if stage == 'train':
            self.transformedTrainData = self.transform(self.trainData)

        elif stage == 'val':
            self.transformedValData = self.transform(self.valData)

        elif stage == 'test':
            self.transformedTestData = self.transform(self.testData)


    def train_dataloader(self):
        return DataLoader(self.transformedTrainData, batch_size=self.batchSize, shuffle=False, num_workers=20)


    def val_dataloader(self):
        return DataLoader(self.transformedTrainData, batch_size=self.batchSize, shuffle=False, num_workers=4)
        # Return DataLoader for Validation Data here


    def test_dataloader(self):
        return DataLoader(self.transformedTrainData, batch_size=self.batchSize, shuffle=False, num_workers=4)
        # Return DataLoader for Testing Data here
