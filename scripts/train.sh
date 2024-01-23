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
num_epochs=200
model_dir=${result_path}'/'${exp_name}'/models'
#----------------------------------------
data_train_dir='/nfs3-p1/hjc/datasets/'${data_name}'/train'
data_test_dir='/nfs3-p1/hjc/datasets/'${data_name}'/test'
#----------------------------------------
log_dir=${result_path}'/runs/'${exp_name}

python engines/train.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --num_epochs ${num_epochs} \
  --model_dir ${model_dir} \
  --data_train_dir ${data_train_dir} \
  --data_test_dir ${data_test_dir} \
  --log_dir ${log_dir}
