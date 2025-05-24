from torch.utils.data import DataLoader
from BBox3D_dataset import BBox3DDataset
from collate import collate_fn
import pytorch_lightning as pl

class BBox3DDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_root,
            batch_size = 1,
            num_workers = 4
    ):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_dataset = BBox3DDataset(
            data_root=self.data_root,
            split='train',
            flip_p = 0.6
        )

        self.val_dataset = BBox3DDataset(
            data_root=self.data_root,
            split='val',
            flip_p = 0.0
        )

        self.test_dataset = BBox3DDataset(
            data_root=self.data_root,
            split='test',
            flip_p = 0.0
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    