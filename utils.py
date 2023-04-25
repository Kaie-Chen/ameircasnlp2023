import os
import csv
import math
import evaluate
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