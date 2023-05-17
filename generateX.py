import csv
import torch
from torch import nn, Tensor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import pipeline
import argparse
import pandas as pd
import numpy

def load_raw_data(src_filepath: str, target_filepath: str = None):
    data = {'src_text': []}
    with open(src_filepath) as f:
        i = 0
        for line in f:
            data['src_text'].append(line.strip())
            i += 1
        print(i)

    if target_filepath:
        data['target_text'] = []
        with open(target_filepath) as f:
            i = 0
            for line in f:
                data['target_text'].append(line.strip())
                i += 1
            print(i)

    return data

def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    return preds
    

class Languages:   
    ### Temperay dataset class until issue 1 fixed 

    def __init__(self, raw_data: Dict[str, List[str]]):
        self.src_text = raw_data['src_text']
        self.target_text = []
        self.with_target = False

        if 'target_text' in raw_data:
            self.with_target = True
            self.target_text = raw_data['target_text']
    
    def __len__(self):
        return len(self.src_text)
    
    def __getitem__(self, idx):
        if self.with_target:
            # for training and validationself.src_text[idx], self.target_text[idx]
            return self.src_text[idx], self.target_text[idx]
        else:
            # for testing
            return self.src_text[idx]


def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    return preds


def predict(model: nn.Module, dataloader: DataLoader, tokenizer: AutoTokenizer,
            device: torch.device, path: str, trglang:str) -> List[List[str]]:
    ### Temperaty                                                                                                                                                          
    model.eval()
    loss_list = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            src_text, trg_text = batch
            inputs = tokenizer(src_text, text_target=trg_text, return_tensors="pt", padding=True).to(device)
            loss, logits = model(**inputs)[:2]
            loss = loss / logits.size()[0]
            loss_list += [loss.cpu().item()]

    return loss_list

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model_path', type=str)
    # args = parser.parse_args()

    model_folder = '/mnt/data3/kaiechen/spanishX/'
    main_folder =  '/mnt/data3/kaiechen/data_generated/'
    file_src_name = 'dedup_filtered.es'
    
    #model                                                                                                                                                                 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    






    #ashaninka language, code cni_Latn        checkpoint-550                                                                                                                          
    print("Processing ashanika")
    ashanika_folder = main_folder + 'ashaninka/'
    print("Model Loading . . . . . . . . . . . . . . . .")
    modelName = model_folder + 'cni_Latn/checkpoint-100'
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    tokenizer = AutoTokenizer.from_pretrained(modelName, src_lang="spa_Latn")
    file_trgt_name = 'dedup_filtered.cni'
    dataCNI = load_raw_data(ashanika_folder + file_src_name, ashanika_folder + file_trgt_name)
    ashaninka_dataloader = DataLoader(Languages(dataCNI), batch_size = 1)
    dataCNI['loss'] = predict(model, ashaninka_dataloader, tokenizer, device, ashanika_folder, "cni_Latn")
    
    data = pd.DataFrame(dataCNI)
    filtered_data = data.nsmallest(2000, ['loss'])
    with open(ashanika_folder + "generated_filtered.es", "w", encoding='utf8') as f:
        for text in filtered_data['src_text']:
            f.write("".join(text) + "\n")
    with open(ashanika_folder + "generated_filtered_1.cni", "w", encoding='utf8') as f:
        for text in filtered_data['target_text']:
            f.write("".join(text) + "\n")
    #predict()                      

     #aymara language, code aym_Latn                                                                                                                                        
    print("Processing aymara")
    aymara_folder = main_folder + 'aymara/'
    modelName = model_folder + 'aym_Latn/checkpoint-1950'
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    file_trgt_name = 'dedup_filtered.aym'
    dataAYM = load_raw_data(aymara_folder + file_src_name, aymara_folder + file_trgt_name)
    aymara_dataloader = DataLoader(Languages(dataAYM), batch_size = 1)
    dataAYM['loss'] = predict(model, aymara_dataloader, tokenizer, device, aymara_folder, "ayr_Latn")
    data = pd.DataFrame(dataAYM)
    filtered_data = data.nsmallest(3000, ['loss'])
    with open(aymara_folder + "generated_filtered.es", "w", encoding='utf8') as f:
        for text in filtered_data['src_text']:
            f.write("".join(text) + "\n")
    with open(aymara_folder + "generated_filtered_1.aym", "w", encoding='utf8') as f:
        for text in filtered_data['target_text']:
            f.write("".join(text) + "\n")

    #bribri language, code bzd_Latn                                                                                                                                        
    # print("Processing bribri")
    # bribri_folder = main_folder + 'bribri/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # bribri_dataloader = DataLoader(Languages(load_raw_data(bribri_folder+'dev.es')), batch_size = 32)
    # predict(model, bribri_dataloader, tokenizer, device, bribri_folder, "bzd_Latn")

    #guarani language, code gn_Latn                                                                                                                                        
    print("Processing guarani")
    guarani_folder = main_folder + 'guarani/'
    modelName = model_folder + 'gn_Latn/checkpoint-1650'
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    file_trgt_name = 'dedup_filtered.gn'
    dataGN = load_raw_data(guarani_folder + file_src_name, guarani_folder + file_trgt_name)
    guarani_dataloader = DataLoader(Languages(dataGN), batch_size = 1)
    dataGN['loss'] = predict(model, guarani_dataloader, tokenizer, device, guarani_folder, "grn_Latn")
    data = pd.DataFrame(dataGN)
    filtered_data = data.nsmallest(4000, ['loss'])
    with open(guarani_folder + "generated_filtered.es", "w", encoding='utf8') as f:
        for text in filtered_data['src_text']:
            f.write("".join(text) + "\n")
    with open(guarani_folder + "generated_filtered_1.gn", "w", encoding='utf8') as f:
        for text in filtered_data['target_text']:
            f.write("".join(text) + "\n")
    

        #hñähñu language, code oto_Latn                                                                                                                                        
    print("Processing hñähñu")
    hñähñu_folder = main_folder + 'hñähñu/'
    modelName = model_folder + 'oto_Latn/checkpoint-800'
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    file_trgt_name = 'dedup_filtered.oto'
    dataOTO = load_raw_data(hñähñu_folder + file_src_name, hñähñu_folder + file_trgt_name)
    hñähñu_dataloader = DataLoader(Languages(dataOTO), batch_size = 1)
    dataOTO['loss'] = predict(model, hñähñu_dataloader, tokenizer, device, hñähñu_folder, "oto_Latn")
    data = pd.DataFrame(dataOTO)
    filtered_data = data.nsmallest(6000, ['loss'])
    with open(hñähñu_folder + "generated_filtered.es", "w", encoding='utf8') as f:
        for text in filtered_data['src_text']:
            f.write("".join(text) + "\n")
    with open(hñähñu_folder + "generated_filtered_1.oto", "w", encoding='utf8') as f:
        for text in filtered_data['target_text']:
            f.write("".join(text) + "\n")

    #nahuatl language, code nah_Latn                                                                                                                                       
    print("Processing nahuatl")
    nahuatl_folder = main_folder + 'nahuatl/'
    modelName = model_folder + 'nah_Latn/checkpoint-1900'
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    file_trgt_name = 'dedup_filtered.nah'
    dataNAH = load_raw_data(nahuatl_folder + file_src_name, nahuatl_folder + file_trgt_name)
    nahuatl_dataloader = DataLoader(Languages(dataNAH), batch_size = 1)
    dataNAH['loss'] = predict(model, nahuatl_dataloader, tokenizer, device, nahuatl_folder, "nah_Latn")
    data = pd.DataFrame(dataNAH)
    filtered_data = data.nsmallest(4000, ['loss'])
    with open(nahuatl_folder + "generated_filtered.es", "w", encoding='utf8') as f:
        for text in filtered_data['src_text']:
            f.write("".join(text) + "\n")
    with open(nahuatl_folder + "generated_filtered_1.nah", "w", encoding='utf8') as f:
        for text in filtered_data['target_text']:
            f.write("".join(text) + "\n")

    #quechua language, code quy_Latn                                                                                                                                       
    # print("Processing quechua")
    # quechua_folder = main_folder + 'quechua/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # quechua_dataloader = DataLoader(Languages(load_raw_data(quechua_folder+'dev.es')), batch_size = 32)
    # predict(model, quechua_dataloader, tokenizer, device, quechua_folder, "quy_Latn")

        #raramuri language, code tar_Latn                                                                                                                                      
    # print("Processing raramuri")
    # raramuri_folder = main_folder + 'raramuri/' + file_name
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # raramuri_dataloader = DataLoader(Languages(load_raw_data(raramuri_folder+'dev.es')), batch_size = 32)
    # predict(model, raramuri_dataloader, tokenizer, device, raramuri_folder, "tar_Latn")

    #shipibo_konibo language, code shp_Latn                                                                                                                                
    print("Processing shipibo_konibo")
    shipibo_konibo_folder = main_folder + 'shipibo_konibo/'
    modelName = model_folder + 'shp_Latn/checkpoint-3300'
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    file_trgt_name = 'dedup_filtered.shp'
    dataSHP = load_raw_data(shipibo_konibo_folder + file_src_name, shipibo_konibo_folder + file_trgt_name)
    shipibo_konibo_dataloader = DataLoader(Languages(dataSHP), batch_size = 1)
    dataSHP['loss'] = predict(model, shipibo_konibo_dataloader, tokenizer, device, shipibo_konibo_folder, "shp_Latn")
    data = pd.DataFrame(dataSHP)
    filtered_data = data.nsmallest(10000, ['loss'])
    with open(shipibo_konibo_folder + "generated_filtered.es", "w", encoding='utf8') as f:
        for text in filtered_data['src_text']:
            f.write("".join(text) + "\n")
    with open(shipibo_konibo_folder + "generated_filtered_1.shp", "w", encoding='utf8') as f:
        for text in filtered_data['target_text']:
            f.write("".join(text) + "\n")

    #wixarika language, code hch_Latn                                                                                                                                      
    print("Processing wixarika")
    wixarika_folder = main_folder + 'wixarika/'
    modelName = model_folder + 'hch_Latn/checkpoint-2750'
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    file_trgt_name = 'dedup_filtered.hch'
    dataHCH = load_raw_data(wixarika_folder + file_src_name, wixarika_folder + file_trgt_name)
    wixarika_dataloader = DataLoader(Languages(dataHCH), batch_size = 1)
    dataHCH['loss'] = predict(model, wixarika_dataloader, tokenizer, device, wixarika_folder, "hch_Latn")
    data = pd.DataFrame(dataHCH)
    filtered_data = data.nsmallest(8000, ['loss'])
    with open(shipibo_konibo_folder + "generated_filtered.es", "w", encoding='utf8') as f:
        for text in filtered_data['src_text']:
            f.write("".join(text) + "\n")
    with open(shipibo_konibo_folder + "generated_filtered_1.hch", "w", encoding='utf8') as f:
        for text in filtered_data['target_text']:
            f.write("".join(text) + "\n")

    print("Done!")

if __name__ == '__main__':
    main()
