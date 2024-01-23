#!/bin/bash
export PYTHONPATH=/nfs3/hjc/projects/cnnlego/code
export CUDA_VISIBLE_DEVICES=0
result_path='/nfs3/hjc/projects/cnnlego/output'
#----------------------------------------
exp_name='lenet_cifar10_base'
#----------------------------------------
#model_name='vgg16'
#model_name='resnet50'
model_name='lenet'
#----------------------------------------
data_name='cifar10'
num_classes=10
#data_name='cifar100'
#num_classes=100
#----------------------------------------
#model_path=${result_path}'/'${exp_name}'/models/model_ori.pth'
model_path=${result_path}'/'${exp_name}'/models/model_disa.pth'
#----------------------------------------
data_dir='/nfs3-p1/hjc/datasets/'${data_name}'/test'
#----------------------------------------

python engines/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir}
