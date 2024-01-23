import os
import argparse
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

import models
import loaders


def partial_conv(conv: nn.Conv2d, inp: torch.Tensor, o_h=None, o_w=None):
    kernel_size = conv.kernel_size
    dilation = conv.dilation
    padding = conv.padding
    stride = conv.stride
    weight = conv.weight.to(inp.device)  # O I K K
    # bias = conv.bias.to(inp.device)  # O

    wei_res = weight.view(weight.size(0), weight.size(1), -1).permute((1, 2, 0))  # I K*K O
    inp_unf = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)(inp)  # B K*K N
    inp_unf = inp_unf.view(inp.size(0), inp.size(1), wei_res.size(1), o_h, o_w)  # B I K*K H_O W_O
    out = torch.einsum('ijkmn,jkl->iljmn', inp_unf, wei_res)  # B O I H W

    # out = out.sum(2)
    # bias = bias.unsqueeze(1).unsqueeze(2).expand((out.size(1), out.size(2), out.size(3)))  # O H W
    # out = out + bias

    return out


def partial_linear(linear: nn.Linear, inp: torch.Tensor):
    weight = linear.weight.to(inp.device)  # (o, i)
    # bias = linear.bias.to(inp.device)  # (o)

    out = torch.einsum('bi,oi->boi', inp, weight)  # (b, o, i)

    # out = torch.sum(out, dim=-1)
    # out = out + bias

    return out


def mm_norm(a, dim=-1, zero=False):
    if zero:
        a_min = torch.zeros(a.size())
    else:
        a_min, _ = torch.min(a, dim=dim, keepdim=True)
    a_max, _ = torch.max(a, dim=dim, keepdim=True)
    a_normalized = (a - a_min) / (a_max - a_min + 1e-5)

    return a_normalized


class HookModule:
    def __init__(self, module):
        self.module = module
        self.inputs = None
        self.outputs = None
        module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, outputs):
        self.inputs = inputs[0]
        self.outputs = outputs


class RelevantFeatureIdentifying:
    def __init__(self, modules, num_classes, save_dir):
        self.modules = [HookModule(module) for module in modules]
        # self.values = [[[] for _ in range(num_classes)] for _ in range(len(modules))]  # [l, c, n, channels]
        self.values = [[0 for _ in range(num_classes)] for _ in range(len(modules))]  # [l, c, channels]
        self.num_classes = num_classes
        self.save_dir = save_dir

    def __call__(self, outputs, labels):
        for layer, module in enumerate(self.modules):
            torch.cuda.empty_cache()
            # print(layer, '==>', layer)
            values = None
            if isinstance(module.module, nn.Conv2d):
                # [b, o, i, h, w]
                values = partial_conv(module.module,
                                      module.inputs,
                                      module.outputs.size(2),
                                      module.outputs.size(3))
                values = torch.sum(values, dim=(3, 4))
            elif isinstance(module.module, nn.Linear):
                # [b, o, i)
                values = partial_linear(module.module,
                                        module.inputs)
            values = torch.relu(values)
            values = values.cpu()
            values = values.numpy()

            for b in range(len(labels)):
                # self.values[layer][labels[b]].append(values[b]) # (l, c, n, o, i)
                self.values[layer][labels[b]] += values[b]  # (l, c, o, i)

    def identify(self):
        # parameter configuration
        alpha_c = 0.3
        beta_c = 0.2
        alpha_f = 0.4
        beta_f = 0.3

        # layer -1
        mask = torch.eye(self.num_classes, dtype=torch.long)  # (c, o)
        mask_path = os.path.join(self.save_dir, 'masks', 'mask_layer{}.pt'.format('-1'))
        torch.save(mask, mask_path)

        # layer 0~n
        for layer, values in enumerate(self.values):  # (l, c, n, o, i)
            values = torch.from_numpy(np.asarray(self.values[layer]))  # (c, n, o, i)
            # values = torch.sum(values, axis=1)  # (c, o, i)
            print('-' * 20)
            print(mask.shape)
            print(values.shape)
            print('-' * 20)

            if values.shape[1] != mask.shape[1]:
                mask = torch.ones((values.shape[0], values.shape[1]), dtype=torch.long)

            values = mm_norm(values)  # (c, o, i)
            if isinstance(self.modules[layer].module, nn.Conv2d):
                values = torch.where(values > alpha_c, 1, 0)  # (c, o, i)
            else:
                values = torch.where(values > alpha_f, 1, 0)  # (c, o, i)
            values = torch.einsum('co,coi->ci', mask, values)  # (c, i)
            # values = torch.sum(values, dim=1)  # (c, i)
            values = mm_norm(values)  # (c, i)
            if isinstance(self.modules[layer].module, nn.Conv2d):
                mask = torch.where(values > beta_c, 1, 0)  # (c, i)
            else:
                mask = torch.where(values > beta_f, 1, 0)  # (c, i)

            mask_path = os.path.join(self.save_dir, 'masks', 'mask_layer{}.pt'.format(layer))
            torch.save(mask, mask_path)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data path')
    parser.add_argument('--save_dir', default='', type=str, help='save dir')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(os.path.join(args.save_dir, 'masks'))
        os.makedirs(os.path.join(args.save_dir, 'figs'))

    print('-' * 50)
    print('TRAIN ON:', device)
    print('DATA DIR:', args.data_dir)
    print('SAVE DIR:', args.save_dir)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    # model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(args.data_dir, args.data_name, data_type='test')

    modules = models.load_modules(model=model)

    rfi = RelevantFeatureIdentifying(modules=modules, num_classes=args.num_classes, save_dir=args.save_dir)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for i, samples in enumerate(tqdm(data_loader)):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            rfi(outputs, labels)

    rfi.identify()


if __name__ == '__main__':
    main()
