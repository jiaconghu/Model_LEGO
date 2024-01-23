from loaders.image_loader import load_images


def load_data(data_dir, data_name, data_type):
    print('-' * 50)
    print('DATA PATH:', data_dir)
    print('DATA NAME:', data_name, '\t|\tDATA TYPE:', data_type)
    print('-' * 50)

    return load_images(data_dir, data_name, data_type)
