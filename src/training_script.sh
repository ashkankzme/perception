CUDA_VISIBLE_DEVICES=2,3 python leave_one_out_training.py leaveOneOutSingle gen &&
CUDA_VISIBLE_DEVICES=3 python leave_one_out_evaluation.py leaveOneOutSingle