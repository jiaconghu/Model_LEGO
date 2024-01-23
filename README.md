# CNN LEGO: Disassembling and Assembling Convolutional Neural Networks

<div align=center><img width="450" src="framework.jpg"/></div>

## Requirements

+ Python version: 3.9
+ PyTorch version: 2.0.1
+ GPU: NVIDIA RTX A6000 / NVIDIA A40

## Quick Start
### Prepare The Source Models

* Training a pre-trained model:
```bash

```


### Model Disassembling
<div align=center><img width="450" src="model_disassembling.jpg"/></div>

* Selecting the Top 1% of Samples with High Confidence:
```bash

```

* Relevant Feature Identifying (\alpha and \beta can be configured in core/relevant_feature_identifying.py):
```bash

```

* Parameter Linking and Model Disassembling (output the disassembled task-aware component):
```bash

```


### Model Assembling
<div align=center><img width="450" src="model_assembling.jpg"/></div>

* Parameter Scaling (optional):
```bash

```

* Alignment Padding and Model Assembling (output the assembled model):
```bash

```

### Others
* Evaluate the accuracy of the model or task-aware component:
```bash

```

* Decision Route Visualization:
```bash

```