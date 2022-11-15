#!/bin/bash
export result_path='/nfs3-p1/hjc/cnnlego/output/'
export exp_name='vgg16_cifar10_10111427'
export model_name='vgg16'
export num_classes=10
export model_path=${result_path}${exp_name}'/models/model_ori.pth'
export locating_path=${result_path}${exp_name}'/locating/'
export disa_path=${result_path}${exp_name}'/models/model_disa.pth'
export disa_layers='-1'
#export disa_layers='0 2 3 5 6 9 10 12 13 15 16 18 19 21 22 24 25 28 29 31 32 34 35 37 38 41 42 44 45 47 48 51 52 53'
#export disa_layers='0 1 2 3 5 6 9 10 11 12 13 15 16 18 19 21 22 24 25 28 29 30 31 32 34 35 37 38 41 42 43 44 45 47 48 51 52'
export disa_labels='0 1 2 3 4'
python core/model_disassemble.py \
  --model_name ${model_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --locating_path ${locating_path} \
  --disa_path ${disa_path} \
  --disa_layers ${disa_layers} \
  --disa_labels ${disa_labels}
