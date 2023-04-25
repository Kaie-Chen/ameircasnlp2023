# LanguageDataset

import os
from os.path import exists
from typing import Dict, List, Optional
from collections import Counter
import csv
import torch
from typing import Dict, List, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class LanguageDataset:   
    def __init__(self, raw_data: Dict[str, List[str]], model_name: str, max_length: int=256):
        self.src_text = raw_data['src_text']
        self.target_text = []
        self.with_target = False
        self.max_length = max_length
        self.model_name = model_name

        if 'target_text' in raw_data:
            self.with_target = True
            self.target_text = raw_data['target_text']

        
    def __len__(self):
        return len(self.src_text)
    

    def __getitem__(self, idx):
        lang_code = self.src_text[idx]['lang_code']
        src_text = self.src_text[idx]['text']
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                  src_lang=lang_code, tgt_lang="spa_Latn")
        input_feature = tokenizer(src_text, max_length=self.max_length, 
                                  padding='max_length', truncation=True)
        if self.with_target:
            # for training and validation
            trg_text = self.target_text[idx]['text']
            label = self.tokenizer(src_text, text_target=trg_text, max_length=self.max_length, 
                                   padding='max_length', truncation=True)
            return label
        else:
            # for testing
            return input_feature