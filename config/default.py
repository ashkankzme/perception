from types import SimpleNamespace


default = SimpleNamespace(
BASE_MODEL_NAME = "google/flan-t5-small",
BATCH_SIZE = 8,
MAX_INPUT_LENGTH = 512,
MAX_OUTPUT_LENGTH = 512,
DATASET_PATH = '../data/trajectories/1_initial/',
MODEL_PATH = '../models/1_initial/',
)

bigDataSmallModel = SimpleNamespace(
BASE_MODEL_NAME = "google/flan-t5-small",
BATCH_SIZE = 8,
MAX_INPUT_LENGTH = 512,
MAX_OUTPUT_LENGTH = 512,
DATASET_PATH = '../data/trajectories/big/',
MODEL_PATH = '../models/bigDataSmallModel/',
)

bigDataMediumModel = SimpleNamespace(
BASE_MODEL_NAME = "google/flan-t5-base",
BATCH_SIZE = 8,
MAX_INPUT_LENGTH = 512,
MAX_OUTPUT_LENGTH = 512,
DATASET_PATH = '../data/trajectories/big/',
MODEL_PATH = '../models/bigDataMediumModel/',
)

bigDataBigModel = SimpleNamespace(
BASE_MODEL_NAME = "google/flan-t5-large",
BATCH_SIZE = 8,
MAX_INPUT_LENGTH = 512,
MAX_OUTPUT_LENGTH = 512,
DATASET_PATH = '../data/trajectories/big/',
MODEL_PATH = '../models/bigDataBigModel/',
)