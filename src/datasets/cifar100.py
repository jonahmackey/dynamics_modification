import os 

import torch
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_and_extract_archive

from src.datasets.classnames import get_classnames


class CIFAR100:
    def __init__(self,
                 preprocess,
                 location,
                 batch_size=128,
                 distributed=False):
        
        self.location = location + '/CIFAR100'
        
        if not os.path.exists(self.location):
            print(f"Dataset not found at {self.location}. Downloading...")
            self.download()
        
        self.test_dataset = datasets.CIFAR100(
            root=self.location,
            download=False,
            train=False,
            transform=preprocess
            )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
            )
        
        self.train_dataset = datasets.CIFAR100(
            root=self.location,
            download=False,
            train=True,
            transform=preprocess
            )
        
        self.train_sampler = None
        if distributed:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset,
                                                                                 shuffle=True,
                                                                                 drop_last=True)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=(self.train_sampler is None),
            drop_last=True,
            num_workers=0,
            sampler=self.train_sampler
        )

        self.classnames = get_classnames('CIFAR100')
        
    def download(self):
        url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
        filename = "cifar-100-python.tar.gz"
        tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
        
        download_and_extract_archive(url, self.location, filename=filename, md5=tgz_md5)