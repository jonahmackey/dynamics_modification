from src.datasets.mnist import MNIST
from src.datasets.svhn import SVHN
from src.datasets.cifar10 import CIFAR10
from src.datasets.cifar100 import CIFAR100

registry = {'MNIST': MNIST, 
            'SVHN': SVHN, 
            'CIFAR10': CIFAR10,
            'CIFAR100': CIFAR100}


def get_dataset(dataset_name, preprocess, location, batch_size=128, distributed=False):
    assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
    
    dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size, distributed=distributed
    )
    return dataset
