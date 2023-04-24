echo "---------ashaninka---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/ashaninka/pretrain_result.txt  --gold_reference ./processed_data/ashaninka/dev.es >> result_new.txt
echo "---------aymara---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/aymara/pretrain_result.txt  --gold_reference ./processed_data/aymara/dev.es >> result_new.txt
echo "---------bribri---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/bribri/pretrain_result.txt  --gold_reference ./processed_data/bribri/dev.es >> result_new.txt
echo "---------guarani---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/guarani/pretrain_result.txt  --gold_reference ./processed_data/guarani/dev.es >> result_new.txt
echo "---------hñähñu---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/hñähñu/pretrain_result.txt  --gold_reference ./processed_data/hñähñu/dev.es >> result_new.txt
echo "---------nahuatl---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/nahuatl/pretrain_result.txt  --gold_reference ./processed_data/nahuatl/dev.es >> result_new.txt
echo "---------quechua---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/quechua/pretrain_result.txt  --gold_reference ./processed_data/quechua/dev.es >> result_new.txt
echo "---------raramuri---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/raramuri/pretrain_result.txt  --gold_reference ./processed_data/raramuri/dev.es >> result_new.txt
echo "---------shipibo_konibo---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/shipibo_konibo/pretrain_result.txt  --gold_reference ./processed_data/shipibo_konibo/dev.es >> result_new.txt
echo "---------wixarika---------" >> result_new.txt
python3 evaluate.py --system_output ./processed_data/wixarika/pretrain_result.txt  --gold_reference ./processed_data/wixarika/dev.es >> result_new.txt
