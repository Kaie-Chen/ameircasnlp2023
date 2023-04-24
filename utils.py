import os
import csv
import math
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Dict, List, Optional


def load_raw_data(src_filepath: str, target_filepath: str = None):
    data = {'src_text': []}
    with open(src_filepath) as f:
        for line in f:
            data['src_text'].append(line.strip())

    if target_filepath:
        data['target_text'] = []
        with open(target_filepath) as f:
            for line in f:
                data['target_text'].append(line.strip())

    return data


def predict(model: nn.Module, dataloader: DataLoader, tokenizer: AutoTokenizer, device: torch.device) -> List[List[str]]:
    ### Temperaty 

    model.eval()
    torch.no_grad()
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = tokenizer(batch, return_tensors="pt").to(device)
            logits = model.generate(**inputs, max_length=512,
                                    forced_bos_token_id=tokenizer.lang_code_to_id["spa_Latn"])

            # only consider non-padded tokens
            # impement later
            
            preds.append(tokenizer.batch_decode(logits, skip_special_tokens=True))
                    
    return preds