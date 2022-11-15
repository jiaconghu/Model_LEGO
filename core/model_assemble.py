import sys

sys.path.append('.')

import torch
import argparse


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model1_path', default='', type=str, help='model path')
    parser.add_argument('--model2_path', default='', type=str, help='model path')
    parser.add_argument('--asse_path', default='', type=str, help='asse path')
    args = parser.parse_args()

    model1 = torch.load(args.model1_path).cuda()
    model2 = torch.load(args.model2_path).cuda()

    # architecture
    print('=================> architecture')
    layer = 0
    for module1, module2 in zip(model1.modules(), model2.modules()):
        if isinstance(module1, torch.nn.Conv2d):
            if layer == 0:
                module1.out_channels += module2.out_channels
            else:
                module1.in_channels += module2.in_channels
                module1.out_channels += module2.out_channels
            layer += 1
        if isinstance(module1, torch.nn.Linear):
            module1.in_features += module2.in_features
            module1.out_features += module2.out_features
        if isinstance(module1, torch.nn.BatchNorm2d):
            module1.num_features += module2.num_features
            module1.running_mean.data = torch.cat([module1.running_mean.data, module2.running_mean.data], dim=0)
            module1.running_var.data = torch.cat([module1.running_var.data, module2.running_var.data], dim=0)
    print(model1)

    # parameter
    print('=================> parameter')
    layer = 0
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if len(p1.shape) > 2:
            if layer == 0:
                p1.data = torch.cat([p1, p2], dim=0)
            else:
                p1b = torch.zeros(p1.shape[0], p2.shape[1], p1.shape[2], p1.shape[2]).cuda()
                p2b = torch.zeros(p2.shape[0], p1.shape[1], p2.shape[2], p2.shape[2]).cuda()
                p1.data = torch.cat([p1, p1b], dim=1)
                p2.data = torch.cat([p2b, p2], dim=1)
                p1.data = torch.cat([p1, p2], dim=0)
            layer += 1
        elif len(p1.shape) > 1:
            p1b = torch.zeros(p1.shape[0], p2.shape[1]).cuda()
            p2b = torch.zeros(p2.shape[0], p1.shape[1]).cuda()
            p1.data = torch.cat([p1, p1b], dim=1)
            p2.data = torch.cat([p2b, p2], dim=1)
            p1.data = torch.cat([p1, p2], dim=0)
        else:
            p1.data = torch.cat([p1, p2], dim=0)
        print('=', p1.shape)

    # save model
    torch.save(model1, args.asse_path)


if __name__ == '__main__':
    main()
