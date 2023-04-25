import os
from os.path import exists
from typing import Dict, List, Optional
from collections import Counter
import csv
import torch
from typing import Dict, List, Optional


class LanguageDataset:   
    def __init__(self, raw_data: Dict[str, List[str]]):
        self.src_text = raw_data['src_text']
        self.target_text = []
        self.with_target = False

        if 'target_text' in raw_data:
            self.with_target = True
            self.target_text = raw_data['target_text']

        
    def __len__(self):
        return len(self.src_text)
    

    def __getitem__(self, idx):
        if self.with_target:
            # for training and validation
            return self.src_text[idx], self.target_text[idx]
        else:
            # for testing
            return self.src_text[idx]