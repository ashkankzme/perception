BASE_MODEL_NAME = "google/flan-t5-small" # better to load model locally
FROZEN_LAYER_DEPTH_THRESHOLD = 1500 # todo this is model dependent, fix it before training
BATCH_SIZE = 8
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
DATASET_PATH = '../data/trajectories/1_initial/'
MODEL_PATH = '../models/1_initial/'