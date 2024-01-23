#!/bin/bash
export PYTHONPATH=/nfs3/hjc/projects/cnnlego/code
export CUDA_VISIBLE_DEVICES=0
#----------------------------------------
mask_dir='/nfs3/hjc/projects/cnnlego/output/lenet_cifar10_base/contributions/masks'
layers='-1'
labels='3 4'
#----------------------------------------

python core/model_decision_route_visualizing.py \
  --mask_dir ${mask_dir} \
  --layers ${layers} \
  --labels ${labels}