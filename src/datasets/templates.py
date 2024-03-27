mnist_template = [
    lambda c: f'a photo of the number: "{c}".',
]



svhn_template = [
    lambda c: f'a photo of the number: "{c}".',
]


dataset_to_template = {
    'MNIST': mnist_template,
    'SVHN': svhn_template,
}


def get_templates(dataset_name):
    assert dataset_name in dataset_to_template, f'Unsupported dataset: {dataset_name}'
    return dataset_to_template[dataset_name]