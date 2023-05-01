import pytorch_lightning as pl
from transformers import AutoTokenizer
from LanguageModelingData import LanugageDataModule
from lightning_transformers.task.nlp.translation import (
    TranslationTransformer,
    WMT16TranslationDataModule,
)
main_folder =  '../processed_data/'
model_name = "facebook/nllb-200-distilled-600M"
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

data = load_raw_data(train_src_filepath, lang_code, model_name, train_trg_filepath, max_length=128)
eval = load_raw_data(eval_src_filepath, lang_code, model_name, eval_trg_filepath, max_length=128)


model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
dm = LanugageDataModule(
    # WMT translation datasets: ['cs-en', 'de-en', 'fi-en', 'ro-en', 'ru-en', 'tr-en']
    data = data,
    eval = eval,
    batch_size = 32,

)
trainer = pl.Trainer(accelerator="gpu", devices="auto", max_epochs=1)

trainer.fit(model, dm)