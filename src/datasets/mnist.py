import torch
import torchvision.datasets as datasets


class MNIST:
    def __init__(self,
                 preprocess,
                 location,
                 batch_size=12,
                 distributed=False):

        self.train_sampler = None
        
        self.test_dataset = datasets.MNIST(
            root=location,
            download=False,
            train=False,
            transform=preprocess
            )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False,
            drop_last=True
            )
        
        self.train_dataset = datasets.MNIST(
            root=location,
            download=False,
            train=True,
            transform=preprocess
            )
        
        if distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            drop_last=True,
            sampler=self.train_sampler
        )

        self.classnames = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']