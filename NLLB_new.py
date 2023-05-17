import csv
import torch
from torch import nn, Tensor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
from transformers import pipeline
import argparse

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


def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    return preds
    

def predict(model: nn.Module, dataloader: DataLoader, tokenizer: AutoTokenizer, device: torch.device, path: str) -> List[List[str]]:
    ### Temperaty 
    model.eval()
    preds = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
            logits = model.generate(**inputs, max_length=512,
                                #tokenizer.lang_code_to_id["quy_Latn"] #grn_Latn
                                    )
            preds += postprocess_text(tokenizer.batch_decode(logits, skip_special_tokens=True))
            
    with open(path+"hch.result.1", "w", encoding='utf8') as f:
        for text in preds:
            f.write("".join(text) + "\n")


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
            return 
        else:
            # for testing
            return self.src_text[idx]


def main():
    """
    LOAD MODEL FROM TERMINAL
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    args = parser.parse_args()
    
    main_folder =  './processed_data/'
    #model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model Loading . . . . . . . . . . . . . . . .")
    modelName = args.model_path
    print(modelName)
    model = AutoModelForSeq2SeqLM.from_pretrained(modelName).to(device)
    print("Model Loaded")

    # #ashaninka language, code cni_Latn number: 256204
    lang_code_to_add = [
            'cni_Latn', 
            #  'aym_Latn', 
            'bzd_Latn', 
            # 'gn_Latn', 
            'oto_Latn', 
            'nah_Latn', 
            # 'quy_Latn', 
            'tar_Latn', 
            'shp_Latn', 
            'hch_Latn'
            ]
    print("Processing")
    ashanika_folder = "./final/"
    tokenizer = AutoTokenizer.from_pretrained(modelName, src_lang="spa_Latn")
    # model.resize_token_embeddings(len(tokenizer))
    ashaninka_dataloader = DataLoader(Languages(load_raw_data("./test.es")), batch_size = 32)
    predict(model, ashaninka_dataloader, tokenizer, device, ashanika_folder)
    
    # #predict()

    #aymara language, code aym_Latn
    # print("Processing aymara")
    # aymara_folder = main_folder + 'aymara/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # aymara_dataloader = DataLoader(Languages(load_raw_data(aymara_folder+'dev.es')), batch_size = 32)
    # predict(model, aymara_dataloader, tokenizer, device, aymara_folder)

    # #bribri language, code bzd_Latn 256205
    # print("Processing bribri")
    # bribri_folder = main_folder + 'bribri/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # bribri_dataloader = DataLoader(Languages(load_raw_data(bribri_folder+'dev.bzd')), batch_size = 32)
    # predict(model, bribri_dataloader, tokenizer, device, bribri_folder)
    
    #guarani language, code gn_Latn  or grn_Latn
    # print("Processing guarani")
    # guarani_folder = main_folder + 'guarani/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # guarani_dataloader = DataLoader(Languages(load_raw_data(guarani_folder+'dev.es')), batch_size = 32)
    # predict(model, guarani_dataloader, tokenizer, device, guarani_folder)
    
    # #hñähñu language, code oto_Latn 256206
    # print("Processing hñähñu")
    # hñähñu_folder = main_folder + 'hñähñu/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # hñähñu_dataloader = DataLoader(Languages(load_raw_data(hñähñu_folder+'dev.es')), batch_size = 32)
    # predict(model, hñähñu_dataloader, tokenizer, device, hñähñu_folder)

    #nahuatl language, code nah_Latn 256207
    # print("Processing nahuatl")
    # nahuatl_folder = main_folder + 'nahuatl/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # nahuatl_dataloader = DataLoader(Languages(load_raw_data(nahuatl_folder+'dev.es')), batch_size = 32)
    # predict(model, nahuatl_dataloader, tokenizer, device, nahuatl_folder)

    # #quechua language, code quy_Latn 
    # print("Processing quechua")
    # quechua_folder = main_folder + 'quechua/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # quechua_dataloader = DataLoader(Languages(load_raw_data(quechua_folder+'dev.quy')), batch_size = 32)
    # predict(model, quechua_dataloader, tokenizer, device, quechua_folder)

    # #raramuri language, code tar_Latn 256208
    # print("Processing raramuri")
    # raramuri_folder = main_folder + 'raramuri/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # raramuri_dataloader = DataLoader(Languages(load_raw_data(raramuri_folder+'dev.tar')), batch_size = 32)
    # predict(model, raramuri_dataloader, tokenizer, device, raramuri_folder)

    #shipibo_konibo language, code shp_Latn 256209
    # print("Processing shipibo_konibo")
    # shipibo_konibo_folder = main_folder + 'shipibo_konibo/'
    # tokenizer = AutoTokenizer.from_pretrained(modelName, src_lang="spa_Latn")
    # shipibo_konibo_dataloader = DataLoader(Languages(load_raw_data(shipibo_konibo_folder+'dev.es')), batch_size = 32)
    # predict(model, shipibo_konibo_dataloader, tokenizer, device, shipibo_konibo_folder)

    #wixarika language, code hch_Latn 256210
    # print("Processing wixarika")
    # wixarika_folder = main_folder + 'wixarika/'
    # tokenizer = AutoTokenizer.from_pretrained( modelName, src_lang="spa_Latn")
    # wixarika_dataloader = DataLoader(Languages(load_raw_data(wixarika_folder+'dev.es')), batch_size = 32)
    # predict(model, wixarika_dataloader, tokenizer, device, wixarika_folder)

    print("Done!")



    
if __name__ == '__main__':
    main()
  





  




