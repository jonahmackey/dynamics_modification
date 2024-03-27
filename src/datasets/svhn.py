import torch
import torchvision.datasets as datasets


class SVHN:
    def __init__(self,
                 preprocess,
                 location,
                 batch_size=128):
        
        modified_location = location + '/svhn'

        self.train_dataset = datasets.SVHN(
            root=modified_location,
            download=False,
            split='train',
            transform=preprocess
            )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
            )

        self.test_dataset = datasets.SVHN(
            root=modified_location,
            download=False,
            split='test',
            transform=preprocess
            )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
            )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']