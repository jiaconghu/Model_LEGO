import torch
from models import simnet, alexnet, vgg, resnet, simplenetv1, googlenet, lenet


def load_model(model_name, in_channels=3, num_classes=10):
    print('-' * 50)
    print('LOAD MODEL:', model_name)
    print('NUM CLASSES:', num_classes)
    print('-' * 50)

    model = None
    if model_name == 'simnet':
        model = simnet.simnet(in_channels, num_classes)
    if model_name == 'alexnet':
        model = alexnet.alexnet(in_channels, num_classes)
    if model_name == 'vgg16':
        model = vgg.vgg16_bn(in_channels, num_classes)
    if model_name == 'resnet50':
        model = resnet.resnet50(in_channels, num_classes)
    if model_name == 'simplenetv1':
        model = simplenetv1.simplenet(in_channels, num_classes)
    if model_name == 'googlenet':
        model = googlenet.googlenet(in_channels, num_classes)
    if model_name == 'lenet':
        model = lenet.lenet(in_channels, num_classes)

    return model


def load_modules(model, model_layers=None):
    assert model_layers is None or type(model_layers) is list

    modules = []
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            modules.append(module)
        if isinstance(module, torch.nn.Linear):
            modules.append(module)
        # if isinstance(module, torch.nn.ReLU):
        #     modules.append(module)

    modules.reverse()  # reverse order
    if model_layers is None:
        model_modules = modules
    else:
        model_modules = []
        for layer in model_layers:
            model_modules.append(modules[layer])

    print('-' * 50)
    print('Model Layers:', model_layers)
    print('Model Modules:', model_modules)
    print('Model Modules Length:', len(model_modules))
    print('-' * 50)

    return model_modules
