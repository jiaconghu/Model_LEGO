#!/bin/bash
export result_path='/nfs3-p1/hjc/cnnlego/output/'
export exp_name='vgg16_cifar10_10111427'
export model1_path=${result_path}${exp_name}'/models/model_disa_1.pth'
export model2_path=${result_path}${exp_name}'/models/model_disa_2.pth'
export asse_path=${result_path}${exp_name}'/models/model_asse.pth'
python core/model_assemble.py \
  --model1_path ${model1_path} \
  --model2_path ${model2_path} \
  --asse_path ${asse_path}
