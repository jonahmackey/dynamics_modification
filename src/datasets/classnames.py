mnist_classes = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
]


svhn_classes = [
    '0',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
]


dataset_to_classnames = {
    'MNIST': mnist_classes,
    'SVHN': svhn_classes,
}


def get_classnames(dataset_name):
    assert dataset_name in dataset_to_classnames, f'Unsupported dataset: {dataset_name}'
    return dataset_to_classnames[dataset_name]