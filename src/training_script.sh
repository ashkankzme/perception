CUDA_VISIBLE_DEVICES=2,3 python train_perception_modeling.py bigDataSmallModel load &&
CUDA_VISIBLE_DEVICES=3 python evaluate_perception_modeling.py bigDataSmallModel &&
CUDA_VISIBLE_DEVICES=2,3 python train_perception_modeling.py bigDataSmallModelLabelsOnly load &&
CUDA_VISIBLE_DEVICES=3 python evaluate_perception_modeling.py bigDataSmallModelLabelsOnly &&
#python train_perception_modeling.py bigDataMediumModelLabelsOnly load &&
#python evaluate_perception_modeling.py bigDataMediumModelLabelsOnly
#python train_perception_modeling.py bigDataLongMediumModel load &&
CUDA_VISIBLE_DEVICES=3 python evaluate_perception_modeling.py bigDataLongMediumModel