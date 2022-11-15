#!/bin/bash
export result_path='/nfs3-p1/hjc/cnnlego/output/'
export exp_name='vgg16_cifar10_10111427'
export model_name='vgg16'
export data_name='cifar10'
export num_classes=10
export num_epochs=200
export model_dir=${result_path}${exp_name}'/models'
export data_dir='/nfs3-p1/hjc/datasets/cifar10'
export log_dir=${result_path}'runs/'${exp_name}
export device_index='3'
python core/train.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --num_epochs ${num_epochs} \
  --model_dir ${model_dir} \
  --data_dir ${data_dir} \
  --log_dir ${log_dir} \
  --device_index ${device_index}
