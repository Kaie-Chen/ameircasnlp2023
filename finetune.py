# finetune.py

import os
import numpy as np
from os.path import exists
from typing import Dict, List, Optional
from collections import Counter
import csv
import torch
from torch import nn, Tensor
import torch.optim as optim
from torch.nn.functional import log_softmax, pad
import math
import copy
import time
from tqdm import tqdm
import torchmetrics
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from torchtext.data.functional import to_map_style_dataset
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
import torchtext.datasets as datasets
import spacy
import GPUtil
import warnings
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
from LanguageDataset import LanguageDataset
from utils import load_raw_data, predict
import gc


def report_gpu():
   print(torch.cuda.list_gpu_processes())
   gc.collect()
   torch.cuda.empty_cache()
report_gpu()


# Load Model
main_folder =  './processed_data/'

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/nllb-200-distilled-600M"

print("Model Loading . . . . . . . . . . . . . . . .")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
print("Model Loaded")


# Load Data
print("Data Loading . . . . . . . . . . . . . . . .")
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="cni_Latn", tgt_lang="spa_Latn")
# Load data 
train_src_filepath = ['ashaninka/dedup_filtered.cni']
train_trg_filepath = ['ashaninka/dedup_filtered.es']
eval_src_filepath = ['ashaninka/dev.cni']
eval_trg_filepath = ['ashaninka/dev.es']
lang_code = ['cni_Latn']

train_raw = load_raw_data(train_src_filepath, lang_code, train_trg_filepath)
eval_raw = load_raw_data(eval_src_filepath, lang_code, eval_trg_filepath)
print("Data Loaded")


# Create dataset
print("Dataset Creating . . . . . . . . . . . . . . . .")
train_data = LanguageDataset(train_raw, model_name, max_length=256)
eval_data = LanguageDataset(eval_raw, model_name, max_length=256)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
print("Dataset Created")

# Copy from huggingface
# Evaluate 
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# Trainer
training_args = Seq2SeqTrainingArguments(
    output_dir="test_finetuned_model",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    warmup_steps=6000,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=20,
    predict_with_generate=True,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Training . . . . . . . . . . . . . . . .")
trainer.train()
print("Trained")