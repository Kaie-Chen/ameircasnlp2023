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


# Load Model
main_folder =  './processed_data/'

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/nllb-200-distilled-600M"

print("Model Loading . . . . . . . . . . . . . . . .")
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
for name, param in model.named_parameters():
    if param.requires_grad and 'decoder' in name:
        param.requires_grad = False

print("Model Loaded")


# Load Data
print("Data Loading . . . . . . . . . . . . . . . .")
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="cni_Latn", tgt_lang="spa_Latn")
# Load data 
train_src_filepath = [main_folder+'ashaninka/dedup_filtered.cni',
                      main_folder+'aymara/dedup_filtered.aym',
                      main_folder+'bribri/dedup_filtered.bzd',
                      main_folder+'guarani/dedup_filtered.gn',
                      main_folder+'hñähñu/dedup_filtered.oto',
                      main_folder+'nahuatl/dedup_filtered.nah',
                      main_folder+'quechua/dedup_filtered.quy',
                      main_folder+'raramuri/dedup_filtered.tar',
                      main_folder+'shipibo_konibo/dedup_filtered.shp',
                      main_folder+'wixarika/dedup_filtered.hch']

train_trg_filepath = [main_folder+'ashaninka/dedup_filtered.es',
                      main_folder+'aymara/dedup_filtered.es',
                      main_folder+'bribri/dedup_filtered.es',
                      main_folder+'guarani/dedup_filtered.es',
                      main_folder+'hñähñu/dedup_filtered.es',
                      main_folder+'nahuatl/dedup_filtered.es',
                      main_folder+'quechua/dedup_filtered.es',
                      main_folder+'raramuri/dedup_filtered.es',
                      main_folder+'shipibo_konibo/dedup_filtered.es',
                      main_folder+'wixarika/dedup_filtered.es']

eval_src_filepath = [main_folder+'ashaninka/dev.cni',
                     main_folder+'aymara/dev.aym',
                     main_folder+'bribri/dev.bzd',
                     main_folder+'guarani/dev.gn',
                     main_folder+'hñähñu/dev.oto',
                     main_folder+'nahuatl/dev.nah',
                     main_folder+'quechua/dev.quy',
                     main_folder+'raramuri/dev.tar',
                     main_folder+'shipibo_konibo/dev.shp',
                     main_folder+'wixarika/dev.hch']

eval_trg_filepath = [main_folder+'ashaninka/dev.es',
                     main_folder+'aymara/dev.es',
                     main_folder+'bribri/dev.es',
                     main_folder+'guarani/dev.es',
                     main_folder+'hñähñu/dev.es',
                     main_folder+'nahuatl/dev.es',
                     main_folder+'quechua/dev.es',
                     main_folder+'raramuri/dev.es',
                     main_folder+'shipibo_konibo/dev.es',
                     main_folder+'wixarika/dev.es']

lang_code = ['cni_Latn', 'aym_Latn', 'bzd_Latn', 'gn_Latn', 'oto_Latn', 
			 'nah_Latn', 'quy_Latn', 'tar_Latn', 'shp_Latn', 'hch_Latn']

train_raw = load_raw_data(train_src_filepath, lang_code, model_name, train_trg_filepath, max_length=256)
eval_raw = load_raw_data(eval_src_filepath, lang_code, model_name, eval_trg_filepath, max_length=256)
print("Data Loaded")


# Create dataset
print("Dataset Creating . . . . . . . . . . . . . . . .")
train_data = LanguageDataset(train_raw)
eval_data = LanguageDataset(eval_raw)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)
del train_raw
del eval_raw
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
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=10,
    num_train_epochs=20,
    predict_with_generate=True,
    fp16=True,
    save_strategy="epoch", 
    load_best_model_at_end=True,
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
