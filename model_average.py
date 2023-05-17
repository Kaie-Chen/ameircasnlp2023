import csv
import torch
from torch import nn, Tensor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import pipeline
import argparse
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AdamW, get_cosine_schedule_with_warmup

print("LOADING MODELS")
#shp_Latn / Done
#cni_Latn / Done
#gn_Latn / Done
#nah_Latn / Done

#hch_Latn / Done
#oto_Latn / Done
#aym_Latn
main_path = '/mnt/data3/kaiechen/spanishX/furtherFine/cni_Latn/'
side_path = '/mnt/data3/kaiechen/spanishX/cni_Latn/'
model_list = [side_path + "checkpoint-250",
              side_path + "checkpoint-200",
              side_path + "checkpoint-150",
              main_path + "checkpoint-1200",
              main_path + "checkpoint-1150",
              ]

state_dict = []
for name in model_list:
    m = AutoModelForSeq2SeqLM.from_pretrained(name)
    state_dict.append(m.state_dict())
print("LOADED")

print("CALCULATING")
sd_average = state_dict[0]
for key in state_dict[0]:
    sd_for_key = float(0)
    for sd in state_dict:
        sd_for_key += sd[key]
    sd_average[key] = sd_for_key / float(len(model_list))
print("CALCULATED")

print("SAVING MODEL")
model = AutoModelForSeq2SeqLM.from_pretrained(model_list[-1])
model.load_state_dict(sd_average)
training_args = Seq2SeqTrainingArguments(
    output_dir="finetune_models",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=8000,
    lr_scheduler_type='constant_with_warmup',
    report_to="wandb",
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=25,
    num_train_epochs=25,
    predict_with_generate=True,
    fp16=True,
    save_strategy="epoch",
    gradient_accumulation_steps=8,
    load_best_model_at_end=True,
    group_by_length=True,
    auto_find_batch_size=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_list[-1], src_lang="cni_Latn", tgt_lang="spa_Latn")
# tokenizer.add_special_tokens({'additional_special_tokens':["cni_Latn", "bzd_Latn", "oto_Latn", "nah_Latn", "tar_Latn", "shp_Latn", "hch_Latn"]})        
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
)
model.resize_token_embeddings(len(tokenizer))
trainer.save_model('/mnt/data3/kaiechen/averaged_withgenerated/cni_Averaged_Latn1')
print("SUCCESS!")