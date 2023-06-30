from types import SimpleNamespace

default = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/1_initial/',
    SAMPLING_RATE=1000,
    MODEL_PATH='../trainedModels/1_initial/',
    DESCRIPTION="Small Data (1000 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

tinyDataSmallModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/tiny/',
    SAMPLING_RATE=100,
    MODEL_PATH='../trainedModels/tinyDataSmallModel/',
    DESCRIPTION="Tiny Data (100 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

smallDataBigBERT = SimpleNamespace(
    BASE_MODEL_NAME="mosaicml/mosaic-bert-base-seqlen-2048",
    TRUST_REMOTE_CODE=True,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=2048,
    MAX_OUTPUT_LENGTH=2048,
    DATASET_PATH='../data/trajectories/1_initial/',
    SAMPLING_RATE=1000,
    MODEL_PATH='../trainedModels/smallDataBigBERT/',
    DESCRIPTION="Small Data (1000 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

bigDataSmallModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/big/',
    MODEL_PATH='../trainedModels/bigDataSmallModel/',
    SAMPLING_RATE=100000,
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

bigDataSmallModelLabelsOnly = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/biglabelsonly/',
    LABELS_ONLY=True,
    MODEL_PATH='../trainedModels/bigDataSmallModelLabelsOnly/',
    SAMPLING_RATE=10000,
    DESCRIPTION="Big Data (10000 Sample Trajectories per worker) including only perception labels, Small Model (flan-t5-small, 77M Parameters)",
)

bigDataMediumModelLabelsOnly = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-base",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=2,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/biglabelsonly/',
    LABELS_ONLY=True,
    MODEL_PATH='../trainedModels/bigDataMediumModelLabelsOnly/',
    SAMPLING_RATE=10000,
    DESCRIPTION="Big Data including only perception labels, Medium Model (flan-t5-base, 250M Parameters)",
)

bigDataMediumModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-base",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/big/',
    SAMPLING_RATE=100000,
    MODEL_PATH='../trainedModels/bigDataMediumModel/',
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Medium Model (flan-t5-base, 250M Parameters)",
)

bigDataBigModel = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-large",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/big/',
    SAMPLING_RATE=100000,
    MODEL_PATH='../trainedModels/bigDataBigModel/',
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Large Model (flan-t5-large, 780M Parameters)",
)

bigDataLongMediumModel = SimpleNamespace(
    BASE_MODEL_NAME="google/long-t5-tglobal-base",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=2,
    ACCUMULATED_BATCHES=16,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/big/',
    MODEL_PATH='../trainedModels/bigDataSmallModel/',
    SAMPLING_RATE=100000,
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)
