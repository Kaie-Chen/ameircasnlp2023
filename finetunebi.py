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
from transformers import AdamW, get_cosine_schedule_with_warmup, get_scheduler
import evaluate
from LanguageDataset import LanguageDataset
from utils import load_raw_data, predict
import gc
import wandb

def main():
    # Load Model
    wandb.login()
    main_folder =  './processed_data/'

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model_name = "finetune_models/checkpoint-16"
    model_name = "facebook/nllb-200-distilled-600M"

    print("Model Loading . . . . . . . . . . . . . . . .")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    for name, param in model.named_parameters():
        if param.requires_grad and ('decoder' in name or 'shared' in name):
            param.requires_grad = False
    print("Model Loaded")


    # Load Data
    print("Data Loading . . . . . . . . . . . . . . . .")
    lang_code = [
                'cni_Latn', 
                'aym_Latn', 
                'bzd_Latn', 
                'gn_Latn', 
                'oto_Latn', 
                'nah_Latn', 
                'quy_Latn', 
                'tar_Latn', 
                'shp_Latn', 
                'hch_Latn'
                ]
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="cni_Latn", tgt_lang="spa_Latn")
#     tokenizer.tgt_lang = 'cni_Latn'
#     #     new_tokens = lang_code
# #     new_tokens = set(new_tokens) - set(tokenizer.vocab.keys())
# #     tokenizer.add_tokens(list(new_tokens))
#     model.resize_token_embeddings(len(tokenizer))
    # Load data 
    train_src_filepath = [
                        main_folder+'ashaninka/dedup_filtered.cni',
                        main_folder+'aymara/dedup_filtered.aym',
                        main_folder+'bribri/dedup_filtered.bzd',
                        main_folder+'guarani/dedup_filtered.gn',
                        main_folder+'hñähñu/dedup_filtered.oto',
                        main_folder+'nahuatl/dedup_filtered.nah',
                        main_folder+'quechua/dedup_filtered.quy',
                        main_folder+'raramuri/dedup_filtered.tar',
                        main_folder+'shipibo_konibo/dedup_filtered.shp',
                        main_folder+'wixarika/dedup_filtered.hch'
                         ]

    train_trg_filepath = [
                        main_folder+'ashaninka/dedup_filtered.es',
                        main_folder+'aymara/dedup_filtered.es',
                        main_folder+'bribri/dedup_filtered.es',
                        main_folder+'guarani/dedup_filtered.es',
                        main_folder+'hñähñu/dedup_filtered.es',
                        main_folder+'nahuatl/dedup_filtered.es',
                        main_folder+'quechua/dedup_filtered.es',
                        main_folder+'raramuri/dedup_filtered.es',
                        main_folder+'shipibo_konibo/dedup_filtered.es',
                        main_folder+'wixarika/dedup_filtered.es'
                         ]

    eval_src_filepath = [
                        main_folder+'ashaninka/dev.cni',
                        main_folder+'aymara/dev.aym',
                        main_folder+'bribri/dev.bzd',
                        main_folder+'guarani/dev.gn',
                        main_folder+'hñähñu/dev.oto',
                        main_folder+'nahuatl/dev.nah',
                        main_folder+'quechua/dev.quy',
                        main_folder+'raramuri/dev.tar',
                        main_folder+'shipibo_konibo/dev.shp',
                        main_folder+'wixarika/dev.hch'
                        ]

    eval_trg_filepath = [
                        main_folder+'ashaninka/dev.es',
                        main_folder+'aymara/dev.es',
                        main_folder+'bribri/dev.es',
                        main_folder+'guarani/dev.es',
                        main_folder+'hñähñu/dev.es',
                        main_folder+'nahuatl/dev.es',
                        main_folder+'quechua/dev.es',
                        main_folder+'raramuri/dev.es',
                        main_folder+'shipibo_konibo/dev.es',
                        main_folder+'wixarika/dev.es'
                        ]

   
    # Copy from huggingface
    # Evaluate 
    metric = evaluate.load("chrf")

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
        result = {"chrf": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    train_data = []
    eval_data = []

#    # for index in range(len(train_src_filepath)):
#     #for index in range(1):
#     train_raw = load_raw_data(train_src_filepath, lang_code, model_name, train_trg_filepath, max_length=256)
#     eval_raw = load_raw_data(eval_src_filepath, lang_code, model_name, eval_trg_filepath, max_length=256)
#     print("Data Loaded")
#     print("Dataset Creating . . . . . . . . . . . . . . . .")
#     train_data = LanguageDataset(train_raw)
#     eval_data = LanguageDataset(eval_raw)
#     print("Dataset Created")
    for i in range(len(train_src_filepath)):
        train_raw = load_raw_data([train_src_filepath[i]],[lang_code[i]], model_name, [train_trg_filepath[i]], max_length=256)
        eval_raw = load_raw_data([eval_src_filepath[i]], [lang_code[i]], model_name, [eval_trg_filepath[i]], max_length=256)
        print("Data Loaded")
        print("Dataset Creating . . . . . . . . . . . . . . . .")
        train_data.append(LanguageDataset(train_raw))
        eval_data.append(LanguageDataset(eval_raw))
        print("Dataset Created")
    

    # Create dataset
   

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model_name)

    del train_raw 
    del eval_raw
    
    languages = [
        "ashaninka",
        "aymara",
        "bribri",
        "guarani",
        "hnahnu",
        "nahuatl",
        "quechua",
        "raramuri",
        "shipibo_konibo",
        "wixarika"
    ]

    
    # print("Training ashaninka . . . . . . . . . . . . . . . .")
    # trainer.train()
    # print("Trained")

    index2 = 0
    for lang in languages:
        # Trainer
        training_args = Seq2SeqTrainingArguments(
            output_dir="/mnt/data2/kaiechen/base" + lang,
            evaluation_strategy="steps",
            report_to="wandb",
            learning_rate=3e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            warmup_steps = 150,
            lr_scheduler_type='cosine',
            weight_decay=0.01,
            save_total_limit=4,
            num_train_epochs=25,
            predict_with_generate=True,
            do_eval = True,
            fp16=True,
            eval_steps = 50,
            save_steps = 50,
            gradient_accumulation_steps=2,
            load_best_model_at_end=True,
            group_by_length=True,
            logging_first_step=True,
            auto_find_batch_size=True,          
            logging_steps=20,
            run_name = lang,)

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data[index2],
            tokenizer=tokenizer,
            eval_dataset=eval_data[index2],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        

        index2 = index2 + 1
        print("Training " + lang + " . . . . . . . . . . . . . . . .")
        trainer.train()
        print("Trained")
        wandb.finish()

        

if __name__ == '__main__':
    main()
