#!/bin/bash
export PYTHONPATH=/nfs3/hjc/projects/cnnlego/code
export CUDA_VISIBLE_DEVICES=0
result_path='/nfs3/hjc/projects/cnnlego/output'
#----------------------------------------
exp_name='lenet_cifar10_base'
#----------------------------------------
model1_path=${result_path}'/'${exp_name}'/models/model_disa1.pth'
model2_path=${result_path}'/'${exp_name}'/models/model_disa2.pth'
asse_path=${result_path}'/'${exp_name}'/models/model_asse.pth'

python core/model_assemble.py \
  --model1_path ${model1_path} \
  --model2_path ${model2_path} \
  --asse_path ${asse_path}
