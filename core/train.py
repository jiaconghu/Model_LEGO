import sys

sys.path.append('.')

import os
import argparse
import time
import shutil
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import loaders
import models
import metrics
from utils.train_util import AverageMeter, ProgressMeter


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--num_epochs', default=200, type=int, help='num epochs')
    parser.add_argument('--model_dir', default='', type=str, help='model dir')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--log_dir', default='', type=str, help='log dir')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_path = os.path.join(args.data_dir, 'train')
    test_path = os.path.join(args.data_dir, 'test')

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    if os.path.exists(args.log_dir):
        shutil.rmtree(args.log_dir)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL DIR:', args.model_dir)
    print('LOG DIR:', args.log_dir)
    print('-' * 50)

    # ----------------------------------------
    # trainer configuration
    # ----------------------------------------
    model = models.load_model(args.model_name, num_classes=args.num_classes)
    model.to(device)

    train_loader = loaders.load_data(args.data_name, train_path, data_type='train')
    test_loader = loaders.load_data(args.data_name, test_path, data_type='test')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=args.num_epochs)

    writer = SummaryWriter(args.log_dir)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    best_acc = None
    best_epoch = None

    for epoch in tqdm(range(args.num_epochs)):
        loss, acc1, acc5 = train(train_loader, model, criterion, optimizer, device)
        writer.add_scalar(tag='training loss', scalar_value=loss.avg, global_step=epoch)
        writer.add_scalar(tag='training acc1', scalar_value=acc1.avg, global_step=epoch)
        loss, acc1, acc5 = test(test_loader, model, criterion, device)
        writer.add_scalar(tag='test loss', scalar_value=loss.avg, global_step=epoch)
        writer.add_scalar(tag='test acc1', scalar_value=acc1.avg, global_step=epoch)

        # ----------------------------------------
        # save best model
        # ----------------------------------------
        if best_acc is None or best_acc < acc1.avg:
            best_acc = acc1.avg
            best_epoch = epoch
            torch.save(model, os.path.join(args.model_dir, 'model_ori.pth'))

        scheduler.step()

    print('COMPLETE !!!')
    print('BEST ACC', best_acc)
    print('BEST EPOCH', best_epoch)
    print('TIME CONSUMED', time.time() - since)


def train(train_loader, model, criterion, optimizer, device):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(train_loader), step=20, prefix='Training',
                             meters=[loss_meter, acc1_meter, acc5_meter])

    model.train()

    for i, samples in enumerate(train_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

        loss_meter.update(loss.item(), inputs.size(0))
        acc1_meter.update(acc1.item(), inputs.size(0))
        acc5_meter.update(acc5.item(), inputs.size(0))

        optimizer.zero_grad()  # 1
        loss.backward()  # 2
        optimizer.step()  # 3

        progress.display(i)

    return loss_meter, acc1_meter, acc5_meter


def test(test_loader, model, criterion, device):
    loss_meter = AverageMeter('Loss', ':.4e')
    acc1_meter = AverageMeter('Acc@1', ':6.2f')
    acc5_meter = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(total=len(test_loader), step=20, prefix='Test',
                             meters=[loss_meter, acc1_meter, acc5_meter])
    model.eval()

    for i, samples in enumerate(test_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc1, acc5 = metrics.accuracy(outputs, labels, topk=(1, 5))

            loss_meter.update(loss.item(), inputs.size(0))
            acc1_meter.update(acc1.item(), inputs.size(0))
            acc5_meter.update(acc5.item(), inputs.size(0))

            progress.display(i)

    return loss_meter, acc1_meter, acc5_meter


if __name__ == '__main__':
    main()
