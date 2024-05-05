import os

import torch
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_and_extract_archive

from src.datasets.classnames import get_classnames


class MNIST:
    def __init__(self,
                 preprocess,
                 location,
                 batch_size=128,
                 distributed=False):
        
        self.location = location
        
        if not os.path.exists(self.location):
            print(f"Dataset not found at {location}. Downloading...")
            self.download()
        
        self.test_dataset = datasets.MNIST(
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
        
        self.train_dataset = datasets.MNIST(
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

        self.classnames = get_classnames('MNIST')
        
    def download(self):
        mirror = "https://ossci-datasets.s3.amazonaws.com/mnist/"

        resources = [
            ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
            ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
            ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
            ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
        ]
        
        os.makedirs(self.location + '/MNIST/raw', exist_ok=True)
        
        for filename, md5 in resources:
            url = f"{mirror}{filename}"
            download_and_extract_archive(url, download_root=self.location + '/MNIST/raw', filename=filename, md5=md5)