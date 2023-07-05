#python train_perception_modeling.py bigDataSmallModel gen &&
python evaluate_perception_modeling.py bigDataSmallModel &&
#python train_perception_modeling.py bigDataSmallModelLabelsOnly gen &&
python evaluate_perception_modeling.py bigDataSmallModelLabelsOnly &&
#python train_perception_modeling.py bigDataMediumModelLabelsOnly load &&
#python evaluate_perception_modeling.py bigDataMediumModelLabelsOnly
#python train_perception_modeling.py bigDataLongMediumModel load &&
python evaluate_perception_modeling.py bigDataLongMediumModel