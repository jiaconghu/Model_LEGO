import sys

sys.path.append('.')

import os
import numpy as np
import argparse

import torch
import torch_pruning as tp

import models


def disassemble():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--locating_path', default='', type=str, help='locating path')
    parser.add_argument('--disa_path', default='', type=str, help='disa path')
    parser.add_argument('--disa_layers', default='', nargs='+', type=int, help='disa layers')
    parser.add_argument('--disa_labels', default='', nargs='+', type=int, help='disa labels')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    locating_path = os.path.join(args.locating_path, 'locating_layer{}.npy')

    print('-' * 50)
    print('RESULT PATH:', args.disa_path)
    print('LOCATING PATH:', locating_path)
    print('-' * 50)

    # ----------------------------------------
    # model configuration
    # ----------------------------------------
    # model = models.load_model(args.model_name, num_classes=args.num_classes)
    # model.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))
    model = torch.load(args.model_path).cpu()
    # print(model.state_dict().keys())

    modules = models.load_modules(model=model, model_layers=None)

    if args.disa_layers[0] == -1:
        args.disa_layers = [i for i in range(len(modules))]
    if args.disa_labels[0] == -1:
        args.disa_labels = [i for i in range(modules[0].out_features)]

    # ----------------------------------------
    # model prune
    # ----------------------------------------
    DG = tp.DependencyGraph()
    DG.build_dependency(model=model, example_inputs=torch.randn(1, 3, 32, 32))

    # layers 0-N input channels
    for layer in args.disa_layers:
        print('---> LAYER', layer)
        print('--->', modules[layer])

        # fn
        prune_fn = None
        if isinstance(modules[layer], torch.nn.Conv2d):
            prune_fn = tp.prune_conv_in_channel
            # prune_fn = tp.prune_conv_out_channel
        if isinstance(modules[layer], torch.nn.Linear):
            prune_fn = tp.prune_linear_in_channel
            # prune_fn = tp.prune_linear_out_channel

        # idxs
        locating = np.load(locating_path.format(layer))
        channel = None
        for label in args.disa_labels:
            if channel is None:
                channel = locating[label]
            else:
                channel = np.logical_or(locating[label], channel)
        idxs = np.where(channel == 0)[0].tolist()
        DG.get_pruning_plan(module=modules[layer], pruning_fn=prune_fn, idxs=idxs).exec()
        print('--->', modules[layer])

    # layer N output channels
    layer = 0
    print('--->', modules[layer])
    prune_fn = tp.prune_linear_out_channel
    locating = np.load(locating_path.format(layer - 1))
    channel = None
    for label in args.disa_labels:
        if channel is None:
            channel = locating[label]
        else:
            channel = np.logical_or(locating[label], channel)
    idxs = np.where(channel == 0)[0].tolist()
    DG.get_pruning_plan(module=modules[layer], pruning_fn=prune_fn, idxs=idxs).exec()
    print('--->', modules[layer])

    # save model
    torch.save(model, args.disa_path)


if __name__ == '__main__':
    disassemble()
