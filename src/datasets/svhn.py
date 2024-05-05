import os

import torch
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_url

from src.datasets.classnames import get_classnames

class SVHN:
    def __init__(self,
                 preprocess,
                 location,
                 batch_size=128,
                 distributed=False):
        
        self.location = location + '/SVHN'
        
        if not os.path.exists(self.location):
            print(f"Dataset not found at {self.location}. Downloading...")
            self.download()
        
        self.test_dataset = datasets.SVHN(
            root=self.location,
            download=False,
            split='test',
            transform=preprocess
            )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
            )
        
        self.train_dataset = datasets.SVHN(
            root=self.location,
            download=False,
            split='train',
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

        self.classnames = get_classnames('SVHN')
    
    def download(self):
        split_list = {
            "train": [
                "http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                "train_32x32.mat",
                "e26dedcc434d2e4c54c9b2d4a06d8373",
                ],
            "test": [
                "http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                "test_32x32.mat",
                "eb5a983be6a315427106f1b164d9cef3",
                ],
            } 
        
        for split in split_list:
            url = split_list[split][0]
            filename = split_list[split][1]
            md5 = split_list[split][2]
            
            download_url(url, self.location, filename, md5)