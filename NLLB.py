import csv
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

def load_raw_data(path: str):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    f.close()
    return data

def predict (tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, outputLang : str, outputFileName : str, data: DataLoader):
    output = []
    for batch in tqdm(data):
        inputs = tokenizer(batch[0], return_tensors="pt")
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[outputLang], max_length=1024)
        output.append(tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0])

    f = open(outputFileName, 'w')
    for e in output:
        f.write(e)
    f.close()

class Languages:
    def __init__(self, raw_data: list):
        self.data = raw_data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    #model 
    print("Model Loading . . . . . . . . . . . . . . . .")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B")
    print("Model Loaded")
    #Spanish language tokenizers
    es_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="es_Latn")

    #ashaninka language, code cni_Latn
    print("Processing ashanika")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="cni_Latn")
    ashaninka_dataloader = DataLoader(Languages(load_raw_data('./processed_data/ashaninka/dev.cni')), batch_size = 1)
    es_ashaninka_dataloader = DataLoader(Languages(load_raw_data('./processed_data/ashaninka/dev.es')), batch_size = 1)
    predict(tokenizer, model, 'es_Latn', './processed_data/ashaninka/predictedAshaninka.txt', ashaninka_dataloader)

    #aymara language, code aym_Latn
    print("Processing aymara")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="aym_Latn")
    aymara_dataloader = DataLoader(Languages(load_raw_data('./processed_data/aymara/dev.aym')), batch_size = 1)
    es_aymara_dataloader = DataLoader(Languages(load_raw_data('./processed_data/aymara/dev.es')), batch_size = 1)

    #bribri language, code bzd_Latn
    print("Processing bribri")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="bzd_Latn")
    bribri_dataloader = DataLoader(Languages(load_raw_data('./processed_data/bribri/dev.bzd')), batch_size = 1)
    es_bribri_dataloader = DataLoader(Languages(load_raw_data('./processed_data/bribri/dev.es')), batch_size = 1)

    #guarani language, code gn_Latn
    print("Processing guarani")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="gn_Latn")
    guarani_dataloader = DataLoader(Languages(load_raw_data('./processed_data/guarani/dev.gn')), batch_size = 1)
    es_guarani_dataloader = DataLoader(Languages(load_raw_data('./processed_data/guarani/dev.es')), batch_size = 1)
    #hñähñu language, code oto_Latn
    print("Processing hñähñu")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="oto_Latn")
    hñähñu_dataloader = DataLoader(Languages(load_raw_data('./processed_data/hñähñu/dev.oto')), batch_size = 1)
    es_hñähñu_dataloader = DataLoader(Languages(load_raw_data('./processed_data/hñähñu/dev.es')), batch_size = 1)

    #nahuatl language, code nah_Latn
    print("Processing nahuatl")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="nah_Latn")
    nahuatl_dataloader = DataLoader(Languages(load_raw_data('./processed_data/nahuatl/dev.nah')), batch_size = 1)
    es_nahuatl_dataloader = DataLoader(Languages(load_raw_data('./processed_data/nahuatl/dev.es')), batch_size = 1)

    #quechua language, code quy_Latn
    print("Processing quechua")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="quy_Latn")
    quechua_dataloader = DataLoader(Languages(load_raw_data('./processed_data/quechua/dev.quy')), batch_size = 1)
    es_quechua_dataloader = DataLoader(Languages(load_raw_data('./processed_data/quechua/dev.es')), batch_size = 1)

    #raramuri language, code tar_Latn
    print("Processing raramuri")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="tar_Latn")
    raramuri_dataloader = DataLoader(Languages(load_raw_data('./processed_data/raramuri/dev.tar')), batch_size = 1)
    es_raramuri_dataloader = DataLoader(Languages(load_raw_data('./processed_data/raramuri/dev.es')), batch_size = 1)

    #shipibo_konibo language, code shp_Latn
    print("Processing shipibo_konibo")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="shp_Latn")
    shipibo_konibo_dataloader = DataLoader(Languages(load_raw_data('./processed_data/shipibo_konibo/dev.shp')), batch_size = 1)
    es_shipibo_konibo_dataloader = DataLoader(Languages(load_raw_data('./processed_data/shipibo_konibo/dev.es')), batch_size = 1)

    #wixarika language, code hch_Latn
    print("Processing wixarika")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", src_lang="hch_Latn")
    wixarika_dataloader = DataLoader(Languages(load_raw_data('./processed_data/wixarika/dev.hch')), batch_size = 1)
    es_wixarika_dataloader = DataLoader(Languages(load_raw_data('./processed_data/wixarika/dev.es')), batch_size = 1)



    
if __name__ == '__main__':
    main()
  




