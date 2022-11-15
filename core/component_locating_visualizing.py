import sys

sys.path.append('.')

import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

import models
import loaders

from utils import image_util
from component_locating import HookModule, partial_conv, partial_linear, norm


class ComponentLocating:
    def __init__(self, modules, num_classes):
        self.modules = [HookModule(module) for module in modules]
        self.values = [[[] for _ in range(num_classes)] for _ in range(len(modules))]  # [l, c, n, channels]
        # self.values = [[0 for _ in range(num_classes)] for _ in range(len(modules))]  # [l, c, channels]
        self.num_classes = num_classes

    def __call__(self, outputs, labels):
        for layer, module in enumerate(self.modules):
            values = None
            if isinstance(module.module, nn.Conv2d):
                # [b, o, i, h, w]
                values = partial_conv(module.module, module.inputs, module.outputs.size(2), module.outputs.size(3))
                values = torch.sum(values, dim=(3, 4))
                values = torch.relu(values)
            elif isinstance(module.module, nn.Linear):
                # [b, o, i]
                values = partial_linear(module.module, module.inputs)
                values = torch.relu(values)

            values = values.detach().cpu().numpy()

            for b in range(len(labels)):
                self.values[layer][labels[b]].append(values[b])  # [l, c, n, o, i]
                # self.values[layer][labels[b]] += values[b]  # [l, c, o, i]

    def sift(self, result_path):
        alpha1 = 0.3
        beta1 = 0.2
        alpha2 = 0.4
        beta2 = 0.3

        # layer -1
        mask = np.zeros((self.num_classes, self.num_classes))  # [c, o]
        for i in range(self.num_classes):
            mask[i][i] = 1
        mask_path = os.path.join(result_path, '{}_layer{}.npy'.format('locating', '-1'))
        np.save(mask_path, mask)

        # layer 0~n
        for layer, values_ in enumerate(self.values):  # [l, c, n, o, i] -> # [c, n, o, i]
            print('---->layer:', layer)
            values_ = np.asarray(values_)  # [c, n, o, i]
            print(values_.shape)

            if layer == 4:
                # ============= single sample =============
                values__ = values_[:, 0:10]  # [c, n, o, i]

                # for label, values in values__:
                values = norm(values__, axis=2, zero_bot=False)  # [c, n, o, i]
                if isinstance(self.modules[layer].module, nn.Conv2d):
                    values = np.where(values > alpha1, 1, 0)  # alpha
                else:
                    values = np.where(values > alpha2, 1, 0)  # alpha
                mask = np.expand_dims(mask, axis=(1, 2)).repeat(10, axis=1)  # [c, o] -> [c, n, 1, o]
                values = np.matmul(mask, values)  # [c, n, 1, i]
                values = np.squeeze(values, axis=1)  # [c, n, i]
                values = norm(values, axis=1, zero_bot=True)  # [c, n, i]

                for label, value in enumerate(values):
                    values_path = os.path.join(result_path, 'v_layer{}_label{}'.format(layer, label))
                    image_util.heatmap(value, values_path, fig_w=256, fig_h=40, annot=False)

                if isinstance(self.modules[layer].module, nn.Conv2d):
                    mask = np.where(values > beta1, 1, 0)
                else:
                    mask = np.where(values > beta2, 1, 0)

            # ============= all samples =============
            values = np.sum(values_, axis=1)  # [c, o, i]

            values = norm(values, axis=2, zero_bot=False)  # [c, o, i]
            if isinstance(self.modules[layer].module, nn.Conv2d):
                values = np.where(values > alpha1, 1, 0)  # alpha
            else:
                values = np.where(values > alpha2, 1, 0)  # alpha
            mask = np.expand_dims(mask, 1)  # [c, 1, o]
            values = np.matmul(mask, values)  # [c, 1, i]
            values = np.squeeze(values, axis=1)  # [c, i]
            values = norm(values, axis=1, zero_bot=True)  # [c, i]

            values_path = os.path.join(result_path, 'v_layer{}'.format(layer))
            image_util.heatmap(values, values_path, fig_w=256, fig_h=40, annot=False)

            if isinstance(self.modules[layer].module, nn.Conv2d):
                mask = np.where(values > beta1, 1, 0)
            else:
                mask = np.where(values > beta2, 1, 0)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--locating_path', default='', type=str, help='locating path')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.locating_path):
        os.makedirs(args.locating_path)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('DATA PATH:', args.data_path)
    print('RESULT PATH:', args.locating_path)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    # model.load_state_dict(torch.load(args.model_path))
    model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(data_name=args.data_name, data_path=args.data_path, data_type='test')

    modules = models.load_modules(model=model)

    component_locating = ComponentLocating(modules=modules, num_classes=args.num_classes)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        component_locating(outputs, labels)

    component_locating.sift(result_path=args.locating_path)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.inf)
    main()
