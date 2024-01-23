#!/bin/bash
export PYTHONPATH=/nfs3/hjc/projects/cnnlego/code
export CUDA_VISIBLE_DEVICES=0
result_path='/nfs3/hjc/projects/cnnlego/output'
#----------------------------------------
exp_name='lenet_cifar10_base'
#----------------------------------------
model_name='lenet'
#----------------------------------------
export data_name='cifar10'
export num_classes=10
#----------------------------------------
export model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
export data_dir=${result_path}'/'${exp_name}'/images/htrain'
export save_dir=${result_path}'/'${exp_name}'/contributions'

python core/relevant_feature_identifying.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
