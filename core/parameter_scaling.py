import collections

import torch
import argparse
import numpy as np
from tqdm import tqdm

import loaders
import models


class ScoreStatistic:
    def __init__(self, num_classes):
        self.scores = [[] for i in range(num_classes)]
        self.nums = torch.zeros(num_classes, dtype=torch.long)

    def __call__(self, outputs, labels):
        scores, predicts = torch.max(outputs.detach(), dim=1)

        for i, label in enumerate(labels):
            if label == predicts[i]:
                self.scores[label].append(scores[i].detach().cpu().numpy())
                self.nums[label] += 1

    def display_score(self, save_path):
        max_num = self.nums.max()
        for i in range(len(self.scores)):
            if len(self.scores[i]) != max_num:
                self.scores[i] = self.scores[i] + [0 for _ in range(max_num - len(self.scores[i]))]
        scores = torch.from_numpy(np.asarray(self.scores))
        scores_class = torch.sum(scores, dim=1) / self.nums
        fc_ratio = self.nums / torch.sum(scores, dim=1)
        np.save(save_path, fc_ratio.numpy())

        print('AVG SCORE RATIO: ', scores_class)
        print('Reciprocal AVG SCORE RATIO: ', fc_ratio)
        print('PICTURE NUM: ', self.nums)
        return fc_ratio


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-' * 100)
    print('SCALE ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA DIR:', args.data_dir)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    state = torch.load(args.model_path)
    if isinstance(state, collections.OrderedDict):
        model = models.load_model(args.model_name)
        model.load_state_dict(state)
    else:
        model = state
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(args.data_dir, args.data_name, data_type='test')

    score_statistic = ScoreStatistic(num_classes=args.num_classes)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for samples in tqdm(data_loader):
        inputs, labels, _ = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        score_statistic(outputs=outputs, labels=labels)

    score_ratio = score_statistic.display_score(
        save_path=args.model_path.split('.')[0] + '.npy')

    # ----------------------------------------
    # parameter scaling
    # ----------------------------------------
    layer = 0
    last_layer = len(models.load_modules(model=model))
    for para in model.parameters():
        if len(para.shape) > 2:  # conv
            layer += 1
        elif len(para.shape) > 1:  # linear
            if layer == last_layer - 1:
                para.data = score_ratio.view(-1, 1).float().cuda() * para.data
            layer += 1
        else:  # bias
            if layer == last_layer:
                para.data = score_ratio.view(-1).float().cuda() * para.data

    scale_model_path = args.model_path.split('.')[0] + '_scale.pth'
    torch.save(model, scale_model_path)

    print('RESCALE MODEL PATH:', scale_model_path)
    print('-' * 50)


if __name__ == '__main__':
    main()
