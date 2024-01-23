# CNN LEGO: Disassembling and Assembling Convolutional Neural Networks

<div align="center"><img width="450" src="framework.jpg"/></div>

## Requirements

+ Python Version: 3.9
+ PyTorch Version: 2.0.1
+ GPU: NVIDIA RTX A6000 / NVIDIA A40

## Quick Start

### Prepare the Source Models

* Train a Pre-trained Model:

```bash
python engines/train.py \
  --model_name 'vgg16' \
  --data_name 'cifar10' \
  --num_classes 10 \
  --num_epochs 200 \
  --model_dir ${model_dir} \
  --data_train_dir ${data_train_dir} \
  --data_test_dir ${data_test_dir} \
  --log_dir ${log_dir}
```

### Model Disassembling

<div align="center"><img width="450" src="model_disassembling.jpg"/></div>

* Select the Top 1% of Samples with High Confidence:

```bash
python core/sample_select.py \
  --model_name 'vgg16' \
  --data_name 'cifar10' \
  --num_classes 10 \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir} \
  --num_samples 50
```

*  Relevant Features Identifying (\alpha and \beta can be configured in core/relevant_feature_identifying.py):

```bash
python core/relevant_feature_identifying.py \
  --model_name 'vgg16' \
  --data_name cifar10 \
  --num_classes 10 \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}
```

* Parameter Linking and Model Assembling (output the disassembled task-aware component):

```bash
python core/model_disassemble.py \
  --model_name 'vgg16' \
  --num_classes 10 \
  --model_path ${model_path} \
  --mask_dir ${mask_dir} \
  --save_dir ${save_dir} \
  --disa_layers ${disa_layers} \
  --disa_labels ${disa_labels}
```

### Model Assembling

<div align="center"><img width="450" src="model_assembling.jpg"/></div>

* Parameter Scaling (optional):

```bash
python core/parameter_scaling.py \
  --model_name 'vgg16' \
  --data_name 'cifar10' \
  --num_classes 10 \
  --model_path ${model_path} \
  --data_dir ${data_dir}
```

* Alignment Padding and Model Assembling (output the assembled model):

```bash
python core/model_assemble.py \
  --model1_path ${model1_path} \
  --model2_path ${model2_path} \
  --asse_path ${asse_path}
```

### Others

* Evaluate the Accuracy of the Model or Task-aware Component:

```bash
python engines/test.py \
  --model_name 'vgg16' \
  --data_name cifar10 \
  --num_classes 10 \
  --model_path ${model_path} \
  --data_dir ${data_dir}
```

* Visualize Model Decision Routes:

```bash
python core/model_decision_route_visualizing.py \
  --mask_dir ${mask_dir} \
  --layers ${layers} \
  --labels ${labels}
```