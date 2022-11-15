import sys

sys.path.append('.')

import os
import torch
import argparse
from tqdm import tqdm

import loaders
import models
from utils import file_util


class SampleSift:
    def __init__(self, num_classes, num_images, is_high_confidence=True):
        self.names = [[None for j in range(num_images)] for i in range(num_classes)]
        self.scores = torch.zeros((num_classes, num_images))
        self.nums = torch.zeros(num_classes, dtype=torch.long)
        self.is_high_confidence = is_high_confidence

    def __call__(self, outputs, labels, names):
        softmax = torch.nn.Softmax(dim=1)(outputs.detach())
        scores, predicts = torch.max(softmax, dim=1)
        # print(scores)

        if self.is_high_confidence:
            for i, label in enumerate(labels):
                if label == predicts[i]:
                    if self.nums[label] == self.scores.shape[1]:
                        score_min, index = torch.min(self.scores[label], dim=0)
                        if scores[i] > score_min:
                            self.scores[label][index] = scores[i]
                            self.names[label.item()][index.item()] = names[i]
                    else:
                        self.scores[label][self.nums[label]] = scores[i]
                        self.names[label.item()][self.nums[label].item()] = names[i]
                        self.nums[label] += 1
        else:
            for i, label in enumerate(labels):
                if self.nums[label] == self.scores.shape[1]:
                    score_max, index = torch.max(self.scores[label], dim=0)
                    if label == predicts[i]:  # TP-LS
                        if scores[i] < score_max:
                            self.scores[label][index] = scores[i]
                            self.names[label.item()][index.item()] = names[i]
                    else:  # TN-HS
                        if -scores[i] < score_max:
                            self.scores[label][index] = -scores[i]
                            self.names[label.item()][index.item()] = names[i]
                else:
                    if label == predicts[i]:  # TP-LS
                        self.scores[label][self.nums[label]] = scores[i]
                        self.names[label.item()][self.nums[label].item()] = names[i]
                        self.nums[label] += 1
                    else:  # TN-HS
                        self.scores[label][self.nums[label]] = -scores[i]
                        self.names[label.item()][self.nums[label].item()] = names[i]
                        self.nums[label] += 1

    def save_image(self, input_path, output_path):
        print(self.scores)
        print(self.nums)

        class_names = sorted([d.name for d in os.scandir(input_path) if d.is_dir()])

        for label, image_list in enumerate(self.names):
            for image in tqdm(image_list):
                class_name = class_names[label]

                src_path = os.path.join(input_path, class_name, str(image))
                dst_path = os.path.join(output_path, class_name, str(image))
                file_util.copy_file(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_path', default='', type=str, help='data path')
    parser.add_argument('--image_path', default='', type=str, help='image path')
    parser.add_argument('--num_images', default=10, type=int, help='num images')
    parser.add_argument('--device_index', default='0', type=str, help='device index')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_index
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.image_path):
        os.makedirs(args.image_path)

    print('-' * 50)
    print('TRAIN ON:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_path)
    print('RESULT PATH:', args.image_path)
    print('-' * 50)

    # ----------------------------------------
    # model/data configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    # model.load_state_dict(torch.load(args.model_path))
    model = torch.load(args.model_path)
    model.to(device)
    model.eval()

    data_loader = loaders.load_data(args.data_name, args.data_path, data_type='test')

    sample_sift = SampleSift(num_classes=args.num_classes, num_images=args.num_images, is_high_confidence=True)

    # ----------------------------------------
    # forward
    # ----------------------------------------
    for samples in tqdm(data_loader):
        inputs, labels, names = samples
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)

        sample_sift(outputs=outputs, labels=labels, names=names)

    sample_sift.save_image(args.data_path, args.image_path)


if __name__ == '__main__':
    main()
