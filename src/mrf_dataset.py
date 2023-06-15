from __future__ import annotations
from torch.utils.data import Dataset
from utils import loadObjectsFromJsonFile
from transformers import AutoTokenizer
import torch


class MRFDataset(Dataset):
    def __init__(self, datasetPath, config):
        super().__init__()
        self.config = config
        self.datasetPath = datasetPath
        self.tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL_NAME)
        self.data = loadObjectsFromJsonFile(self.datasetPath)
        # self.data = MRFDataset.formatInput(self.data)
        self.data = self.transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def transform(self, rawDataset):

        inputs = [dp['X'] for dp in rawDataset]
        outputs = [dp['y'] for dp in rawDataset]

        model_inputs = self.tokenizer(inputs, max_length=self.config.MAX_INPUT_LENGTH, padding="max_length", truncation=True)

        # encode the summaries
        labels = self.tokenizer(outputs, max_length=self.config.MAX_OUTPUT_LENGTH, padding="max_length", truncation=True).input_ids

        # important: we need to replace the index of the padding tokens by -100
        # such that they are not taken into account by the CrossEntropyLoss
        labels_with_ignore_index = []
        for labels_example in labels:
            labels_example = [label if label != 0 else -100 for label in labels_example]
            labels_with_ignore_index.append(labels_example)

        # model_inputs["labels"] = labels_with_ignore_index

        model_inputs = [{
            "input_ids": torch.tensor(model_inputs.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(model_inputs.attention_mask[idx], dtype=torch.long),
            "labels": torch.tensor(labels_with_ignore_index[idx], dtype=torch.long),
        } for idx in range(len(model_inputs.input_ids))]

        return model_inputs