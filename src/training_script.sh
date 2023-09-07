CUDA_VISIBLE_DEVICES=1,2,3 python hot_start_cv_training.py HotStartCV gen &&
CUDA_VISIBLE_DEVICES=3 python hot_start_cv_evaluation.py HotStartCV