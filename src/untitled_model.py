import torch
import torch.nn as nn
from src.nanoGPT import GPT
import config.default as defaultConfig
from src.nanoGPT.config import train_gpt2 as gpt2Config
from sentence_transformers import SentenceTransformer



# this class has a decision transformer and an SBERT embedding model
class UntitledModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.decisionTransformer = GPT(gpt2Config) # todo tune GPT config for my task, right now using default GPT2 config

        self.textEncoder = SentenceTransformer(defaultConfig.SBERT_MODEL_NAME) # todo should we load model here?
        # freeze all layers after a certain depth, as determined by defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD
        for param in self.textEncoder.parameters()[:defaultConfig.FROZEN_LAYER_DEPTH_THRESHOLD]: # todo test
            param.requires_grad = False


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        pass