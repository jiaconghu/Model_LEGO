import os
from torch.utils.data import DataLoader

from torchvision import transforms
from loaders.datasets import ImageDataset


def _get_train_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.RandomHorizontalFlip(0.5),
                            transforms.RandomVerticalFlip(0.5),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])

                        ]))


def _get_test_set(data_path):
    return ImageDataset(image_dir=data_path,
                        transform=transforms.Compose([
                            transforms.Resize((64, 64)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])
                        ]))


def load_images(data_path, data_type=None):
    assert data_type is None or data_type in ['train', 'test']
    if data_type == 'train':
        data_set = _get_train_set(data_path)
    else:
        data_set = _get_test_set(data_path)

    data_loader = DataLoader(dataset=data_set,
                             batch_size=128,
                             num_workers=8,
                             shuffle=True)

    return data_loader
