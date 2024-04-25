import torch
import torchvision.datasets as datasets
from src.datasets.classnames import get_classnames

class SVHN:
    def __init__(self,
                 preprocess,
                 location,
                 batch_size=128,
                 distributed=False):
        
        location = location + '/SVHN'

        test_batch_size = batch_size
        if distributed:
            test_batch_size = 32
        
        self.test_dataset = datasets.SVHN(
            root=location,
            download=False,
            split='test',
            transform=preprocess
            )

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            drop_last=True
            )
        
        self.train_dataset = datasets.SVHN(
            root=location,
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