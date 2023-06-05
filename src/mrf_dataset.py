from __future__ import annotations
from torch.utils.data import Dataset
from utils import loadObjectsFromJsonFile
from transformers import T5Tokenizer
import torch


# remove from here
import sys
sys.path.insert(0, '/local2/ashkank/perception/config/')
import default as defaultConfig


class MRFDataset(Dataset):
    def __init__(self, datasetPath):
        super().__init__()
        self.datasetPath = datasetPath
        self.tokenizer = T5Tokenizer.from_pretrained(defaultConfig.BASE_MODEL_NAME)
        self.data = loadObjectsFromJsonFile(self.datasetPath)
        self.data = self.formatInput(self.data)
        self.data = self.transform(self.data)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def formatInput(self, inputs):
        formatedInput = []
        for i, trajectory in enumerate(inputs):
            dataPoint = {'X': '', 'y': ''}
            formattedTrajectory = '<header> ' + trajectory['header'] + ' </header>'
            for j, frame in enumerate(trajectory['inputFrames']):
                formattedTrajectory += f' <frame_{j}> ' + frame + ' </frame_{j}>'

            formattedTrajectory += ' <query> ' + trajectory['query'] + ' </query>'
            dataPoint['X'] = formattedTrajectory
            dataPoint['y'] = trajectory['prediction']
            formatedInput.append(dataPoint)

        return formatedInput

    def transform(self, rawDataset):

        inputs = [dp['X'] for dp in rawDataset]
        outputs = [dp['y'] for dp in rawDataset]

        model_inputs = self.tokenizer(inputs, max_length=defaultConfig.MAX_INPUT_LENGTH, padding="max_length", truncation=True)

        # encode the summaries
        labels = self.tokenizer(outputs, max_length=defaultConfig.MAX_OUTPUT_LENGTH, padding="max_length", truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        # model_inputs["labels"] = labels_with_ignore_index
        model_inputs = [{
            "input_ids": torch.as_tensor(model_inputs.input_ids[idx]),
            "attention_mask": torch.as_tensor(model_inputs.attention_mask[idx]),
            "labels": torch.as_tensor(labels_with_ignore_index[idx])
        } for idx in range(len(model_inputs.input_ids))]

        return model_inputs