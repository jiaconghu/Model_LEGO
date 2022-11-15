from loaders.cifar_loader import load_images as load_cifar
from loaders.tiny_imagenet_loader import load_images as load_tiny_imagenet
from loaders.mnist_loader import load_images as load_mnist


def load_data(data_name, data_path, data_type=None):
    print('-' * 50)
    print('DATA NAME:', data_name)
    print('DATA PATH:', data_path)
    print('DATA TYPE:', data_type)
    print('-' * 50)

    assert data_name in ['mnist', 'fashion-mnist', 'cifar10', 'cifar100', 'tiny_imagenet']

    data_loader = None
    if data_name == 'mnist' or data_name == 'fashion-mnist':
        data_loader = load_mnist(data_path, data_type)
    elif data_name == 'cifar10' or data_name == 'cifar100':
        data_loader = load_cifar(data_path, data_type)
    elif data_name == 'tiny_imagenet':
        data_loader = load_tiny_imagenet(data_path, data_type)
    return data_loader
