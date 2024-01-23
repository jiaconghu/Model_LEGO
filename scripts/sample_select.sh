#!/bin/bash
export PYTHONPATH=/nfs3/hjc/projects/cnnlego/code
export CUDA_VISIBLE_DEVICES=0
result_path='/nfs3/hjc/projects/cnnlego/output'
#----------------------------------------
exp_name='lenet_cifar10_base'
#----------------------------------------
model_name='lenet'
#----------------------------------------
data_name='cifar10'
num_classes=10
#----------------------------------------
model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
data_dir='/nfs3-p1/hjc/datasets/cifar10/train'
save_dir=${result_path}'/'${exp_name}'/images/htrain'
num_samples=50

python core/sample_select.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir} \
  --num_samples ${num_samples}
