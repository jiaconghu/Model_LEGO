import os
from torch.utils.data import DataLoader

from torchvision import transforms
from loaders.datasets import ImageDataset


def _get_train_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010)),
                        ]))


def _get_test_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((32, 32)),
                            transforms.ToTensor(),
                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                 (0.2023, 0.1994, 0.2010)),
                        ]))


def load_images(data_path, data_type=None):
    assert data_type is None or data_type in ['train', 'test']
    if data_type == 'train':
        data_set = _get_train_set(data_path)
    else:
        data_set = _get_test_set(data_path)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=4,
                             shuffle=True)

    return data_loader
