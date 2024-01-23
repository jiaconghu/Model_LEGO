import os
import argparse
import time

import torch
from torch import nn
import collections

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter

from thop import profile


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data directory')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-' * 50)
    print('TEST ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    state = torch.load(args.model_path)
    if isinstance(state, collections.OrderedDict):
        model = models.load_model(args.model_name, num_classes=args.num_classes)
        model.load_state_dict(state)
    else:
        model = state
    model.to(device)

    test_loader = loaders.load_data(args.data_dir, args.data_name, data_type='test')

    criterion = nn.CrossEntropyLoss()

    # ----------------------------------------
    # speed
    # ----------------------------------------
    speed(model, device)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    # since = time.time()

    loss, acc1, acc5, class_acc = test(test_loader, model, criterion, device)

    print('-' * 50)
    print(class_acc)
    print('AVG:', acc1.avg)
    # print('TIME CONSUMED', time.time() - since)


def test(test_loader, model, criterion, device):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
                             meters=[loss_meter, acc1_meter, acc5_meter])
    class_acc = metrics.ClassAccuracy()
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 1))
            class_acc.update(outputs, labels)

            loss_meter.update(loss.item(), inputs.size(0))
            acc1_meter.update(acc1.item(), inputs.size(0))
            acc5_meter.update(acc5.item(), inputs.size(0))

            progress.display(i)

    return loss_meter, acc1_meter, acc5_meter, class_acc


def speed(model, device):
    # model.eval()

    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32).to(device),))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


if __name__ == '__main__':
    main()
