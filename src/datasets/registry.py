from src.datasets.mnist import MNIST
from src.datasets.svhn import SVHN

registry = {'MNIST': MNIST, 'SVHN': SVHN}


def get_dataset(dataset_name, preprocess, location, batch_size=128):
    assert dataset_name in registry, f'Unsupported dataset: {dataset_name}. Supported datasets: {list(registry.keys())}'
    
    dataset_class = registry[dataset_name]
    dataset = dataset_class(
        preprocess, location=location, batch_size=batch_size
    )
    return dataset
