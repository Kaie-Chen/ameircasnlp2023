import os
from os.path import exists
from typing import Dict, List, Optional
from collections import Counter
import csv
import torch
from typing import Dict, List, Optional


class LanguageDataset:   
    def __init__(self, raw_data: Dict[str, List[str]], tokenizer, max_length=256):
        self.src_text = raw_data['src_text']
        self.target_text = []
        self.with_target = False
        self.tokenizer = tokenizer
        self.max_length = max_length

        if 'target_text' in raw_data:
            self.with_target = True
            self.target_text = raw_data['target_text']

        
    def __len__(self):
        return len(self.src_text)
    

    def __getitem__(self, idx):
        input_feature = self.tokenizer(self.src_text[idx], max_length=self.max_length, 
                                           padding='max_length', truncation=True)
        if self.with_target:
            label = self.tokenizer(self.src_text[idx], text_target=self.target_text[idx], max_length=self.max_length, 
                                       padding='max_length', truncation=True)
            # for training and validation
            return label
        else:
            # for testing
            return input_feature