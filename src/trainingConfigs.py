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
    DESCRIPTION="Including only perception labels, No demopgrahics, Small Model (flan-t5-small, 77M Parameters)",
    MASKED_DEMOGRAPHICS=True,
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
    BATCH_SIZE=1,
    ACCUMULATED_BATCHES=32,
    MAX_INPUT_LENGTH=1024,
    MAX_OUTPUT_LENGTH=128,
    DATASET_PATH='../data/trajectories/big/',
    MODEL_PATH='../trainedModels/bigDataLongMediumModel/',
    SAMPLING_RATE=100000,
    DESCRIPTION="Big Data (100000 Sample Trajectories per worker), Small Model (flan-t5-small, 77M Parameters)",
)

leaveOneOut = SimpleNamespace(
    BASE_MODEL_NAME="google/flan-t5-small",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/leaveoneout/',
    LABELS_ONLY=True,
    MODEL_PATH='../trainedModels/leaveOneOut',
    SAMPLING_RATE=10000,
    DESCRIPTION="Leave one out training/evaluation for users with demographics, on a base model trained on all other users w/out demographics",
    MASKED_DEMOGRAPHICS=False,
)

HotStartCV = SimpleNamespace(
    BASE_MODEL_NAME='../trainedModels/leaveOneOut_loo_',
    TOKENIZER_NAME="google/flan-t5-small",
    TRUST_REMOTE_CODE=False,
    BATCH_SIZE=4,
    MAX_INPUT_LENGTH=512,
    MAX_OUTPUT_LENGTH=512,
    DATASET_PATH='../data/trajectories/perworker/',
    LABELS_ONLY=True,
    MODEL_PATH='../trainedModels/10fold_cvModels_new/hotStartCV',
    SAMPLING_RATE=10000,
    DESCRIPTION="Hot Start, cross validation training/evaluation, built on top of leave one out models",
    MASKED_DEMOGRAPHICS=False,
    NUM_TRAIN_EPOCHS=1,
    WARMUP_STEPS=5,
    FOLDS=10,
)