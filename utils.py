# utils.py

import os
import csv
import math
import evaluate
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, List, Optional


def load_raw_data(src_filepath: List[str], lang_code: List[str], trg_filepath: List[str] = None):
    len_src = len(src_filepath)
    if len_src != len(lang_code):
        raise Exception("Lengths of src_filepath and lang_code don't match.")
    
    data = {'src_text': []}
    for i in range(len_src):
        path = src_filepath[i]
        code = lang_code[i]

        with open(path) as f:
            for line in f:
                data['src_text'].append({'text': line.strip(), 'lang_code': code})

    if trg_filepath:
        data['target_text'] = []
        if len_src != len(trg_filepath):
            raise Exception("Lengths of src_filepath and trg_filepath don't match.")
        
        for i in range(len_src):
            path = trg_filepath[i] 
            code = lang_code[i]

            with open(path) as f:
                for line in f:
                    data['target_text'].append({'text': line.strip(), 'lang_code': lang_code})

    return data


def predict(
    model: nn.Module, 
    dataloader: DataLoader, 
    tokenizer: AutoTokenizer, 
    device: torch.device
) -> List[List[int]]:
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = batch.to(device)

            logits = model.generate(**inputs, max_length=256,
                                    forced_bos_token_id=tokenizer.lang_code_to_id["spa_Latn"])
            
            preds.append(logits)
                    
    return preds