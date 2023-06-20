from types import SimpleNamespace

default = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    BATCH_SIZE=8,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/1_initial/',
    SAMPLING_RATE='1000',
    MODEL_PATH='../models/1_initial/',
    DESCRIPTION="Small Data (1000 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

tinyDataSmallModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    BATCH_SIZE=8,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/tiny/',
    SAMPLING_RATE='100',
    MODEL_PATH='../models/tinyDataSmallModel/',
    DESCRIPTION="Tiny Data (100 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

bigDataSmallModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    BATCH_SIZE=8,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/big/',
    MODEL_PATH='../models/bigDataSmallModel/',
    SAMPLING_RATE='100000',
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

bigDataMediumModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-base",
    BATCH_SIZE=8,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/big/',
    SAMPLING_RATE='100000',
    MODEL_PATH='../models/bigDataMediumModel/',
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Medium Model (flan-t5-base, 250M Parameters)",
)

bigDataBigModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-large",
    BATCH_SIZE=8,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/big/',
    SAMPLING_RATE='100000',
    MODEL_PATH='../models/bigDataBigModel/',
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Large Model (flan-t5-large, 780M Parameters)",
)
