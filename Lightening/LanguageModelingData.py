from torch.utils.data import random_split,  DataLoader
import pytorch_lightning as pl
from typing import Optional
from utils import load_raw_data, predict
from LanguageDataSet import LanguageDataset
#from dataset import load_dataset
#import LanguageDataset

class LanugageDataModule(pl.LightningDataModule):
    """
    DataModule used for semantic segmentation in geometric generalization project
    """

    def __init__(self,data, eval, batch_size, numOfWorker = 8):
        self.data = data
        self.batch_size = batch_size
        self.model_name = model_name
        self.numOfWorker = numOfWorker
      
        
    def prepare_data(self):
        """
        Empty prepare_data method left in intentionally. 
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#prepare-data
        """
        pass
    def setup(self, stage: Optional[str] = None):
        """
        Method to setup your datasets, here you can use whatever dataset class you have defined in Pytorch and prepare the data in order to pass it to the loaders later
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html#setup
        """
      
        # Assign train/val datasets for use in dataloaders
        # the stage is used in the Pytorch Lightning trainer method, which you can call as fit (training, evaluation) or test, also you can use it for predict, not implemented here
        
        if stage == "fit" or stage is None:
            train_set_full =  LanguageDataset(self.data)
            train_set_size = int(len(train_set_full) * 0.9)
            valid_set_size = len(train_set_full) - train_set_size
            self.train, self.validate = random_split(train_set_full, [train_set_size, valid_set_size])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.test = LanguageDataset()
            
    # define your dataloaders
    # again, here defined for train, validate and test, not for predict as the project is not there yet. 
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True, num_workers=self.numOfWorker)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.numOfWorker)

    def test_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size, num_workers=self.numOfWorker)