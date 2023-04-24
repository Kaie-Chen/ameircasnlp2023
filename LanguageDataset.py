import os
from os.path import exists
from typing import Dict, List, Optional
from collections import Counter
import csv
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, List, Optional


class Tokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        token_ids = self.tokenizer(text)['input_ids']
        if max_length:
            if max_length < len(token_ids): 
                token_ids = token_ids[:max_length]
            else:
                token_ids.extend([-1e4 for i in range(max_length-len(token_ids))])
        
        return token_ids


class LanguageDataset:   
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
        src_output = {}
        input_ids = torch.LongTensor(self.src_tokenizer.encode(self.src_text[idx], max_length=256))
        attention_mask = input_ids > -1e4
        if self.with_target:
            # for training and validation
            return input_ids, attention_mask.float(), torch.LongTensor.encode(self.target_tokenizer(self.target_text[idx], max_length=256))
        else:
            # for testing
            return input_ids, attention_mask.float()