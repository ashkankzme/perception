import torch
import torch.nn as nn
import config.default as defaultConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration


# this class has a decision transformer and an SBERT embedding model
class MisinfoPerceptionT5(nn.Module):
    def __init__(self, trainingConfig):
        super().__init__()

        self.tokenizer = T5Tokenizer.from_pretrained(defaultConfig.BASE_MODEL_NAME)
        self.model = T5ForConditionalGeneration.from_pretrained(defaultConfig.BASE_MODEL_NAME, device_map="auto")  # check whether the device_map config makes sense
        self.trainingConfig = trainingConfig

        # freeze all layers after a certain depth, as determined by defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD
        for param in self.model.parameters()[:defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD]:  # todo test
            param.requires_grad = False





    def forward(self, inputPrompt):
        input_ids = self.tokenizer(inputPrompt, return_tensors="pt").input_ids.to("cuda")  # is this parallelizable?

        outputs = self.model.generate(input_ids)
        print(self.tokenizer.decode(outputs[0]))
