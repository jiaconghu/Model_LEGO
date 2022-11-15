#!/bin/bash
export result_path='/nfs3-p1/hjc/cnnlego/output/'
export exp_name='vgg16_cifar10_10111427'
export model_name='vgg16'
export data_name='cifar10'
export num_classes=10
#export model_path=${result_path}${exp_name}'/models/model_ori.pth'
#export model_path=${result_path}${exp_name}'/models/model_disa.pth'
export model_path=${result_path}${exp_name}'/models/model_asse.pth'
export data_path='/nfs4-p1/gj/datasets/cifar10/test'
export device_index='1'
python core/test.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_path ${data_path} \
  --device_index ${device_index}
