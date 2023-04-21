import os
from os.path import exists
from typing import Dict, List, Optional
from collections import Counter
import csv
import torch

class LanguageDataset:   
    """
    Not Awailable
    """

    def __init__(self, raw_data: Dict[str, List[str]], src_tokenizer: AutoTokenizer, target_tokenizer: AutoTokenizer = None):
        self.src_tokenizer = src_tokenizer
        self.src_text = raw_data['src_text']
        self.target_text = []
        self.with_target = False

        if 'target_text' in raw_data:
            if not target_tokenizer:
                raise Exception("Expect tokenizer for target language: Got None")

            self.with_target = True
            self.target_tokenizer = target_tokenizer
            self.target_text = raw_data['target_text']

        
    def __len__(self):
        return len(self.src_text)
    

    def __getitem__(self, idx):
        src_token = self.src_tokenizer(self.src_text[idx], return_tensors="pt")
        if self.with_target:
            # for training and validation
            return src_token, self.target_tokenizer(self.target_text[idx], return_tensors="pt")
        else:
            # for testing
            return src_token