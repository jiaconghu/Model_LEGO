from torch.utils.data import DataLoader
from torchvision import transforms
from loaders.datasets import ImageDataset

mnist_train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

mnist_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5)),
])

cifar10_train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize((32, 32)),
    # transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                      (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                         (0.24703233, 0.24348505, 0.26158768)),
])

cifar10_test_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    # transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465),
    #                      (0.2023, 0.1994, 0.2010)),
    transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                         (0.24703233, 0.24348505, 0.26158768)),
])

tiny_imagenet_train_transform = transforms.Compose([
    transforms.RandomResizedCrop((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975),
                         (0.2770, 0.2691, 0.2821))
])

tiny_imagenet_test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975),
                         (0.2770, 0.2691, 0.2821))
])

imagenet_train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

imagenet_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])


def _get_set(data_path, transform):
    return ImageDataset(image_dir=data_path,
                        transform=transform)


def load_images(data_dir, data_name, data_type=None):
    assert data_name in ['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'tiny-imagenet', 'imagenet']
    assert data_type is None or data_type in ['train', 'test']

    data_transform = None
    if data_name == 'mnist' and data_type == 'train':
        data_transform = mnist_train_transform
    elif data_name == 'mnist' and data_type == 'test':
        data_transform = mnist_test_transform
    elif data_name == 'cifar10' and data_type == 'train':
        data_transform = cifar10_train_transform
    elif data_name == 'cifar10' and data_type == 'test':
        data_transform = cifar10_test_transform
    elif data_name == 'cifar100' and data_type == 'train':
        data_transform = cifar10_train_transform
    elif data_name == 'cifar100' and data_type == 'test':
        data_transform = cifar10_test_transform
    elif data_name == 'tiny-imagenet' and data_type == 'train':
        data_transform = tiny_imagenet_train_transform
    elif data_name == 'tiny-imagenet' and data_type == 'test':
        data_transform = tiny_imagenet_test_transform
    elif data_name == 'imagenet' and data_type == 'train':
        data_transform = imagenet_train_transform
    elif data_name == 'imagenet' and data_type == 'test':
        data_transform = imagenet_test_transform
    assert data_transform is not None

    data_set = _get_set(data_dir, transform=data_transform)
    data_loader = DataLoader(dataset=data_set,
                             batch_size=256,
                             num_workers=4,
                             shuffle=True)
# ImageNet+VGG16: bs128->gpu26311->40days
    return data_loader
