echo "---------ashaninka---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/ashaninka/finetune_result.txt  --gold_reference ./processed_data/ashaninka/dev.es >> result_finetune.txt
echo "---------aymara---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/aymara/finetune_result.txt  --gold_reference ./processed_data/aymara/dev.es >> result_finetune.txt
echo "---------bribri---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/bribri/finetune_result.txt  --gold_reference ./processed_data/bribri/dev.es >> result_finetune.txt
echo "---------guarani---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/guarani/finetune_result.txt  --gold_reference ./processed_data/guarani/dev.es >> result_finetune.txt
echo "---------hñähñu---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/hñähñu/finetune_result.txt  --gold_reference ./processed_data/hñähñu/dev.es >> result_finetune.txt
echo "---------nahuatl---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/nahuatl/finetune_result.txt  --gold_reference ./processed_data/nahuatl/dev.es >> result_finetune.txt
echo "---------quechua---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/quechua/finetune_result.txt  --gold_reference ./processed_data/quechua/dev.es >> result_finetune.txt
echo "---------raramuri---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/raramuri/finetune_result.txt  --gold_reference ./processed_data/raramuri/dev.es >> result_finetune.txt
echo "---------shipibo_konibo---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/shipibo_konibo/finetune_result.txt  --gold_reference ./processed_data/shipibo_konibo/dev.es >> result_finetune.txt
echo "---------wixarika---------" >> result_finetune.txt
python3 evaluate_result.py --system_output ./processed_data/wixarika/finetune_result.txt  --gold_reference ./processed_data/wixarika/dev.es >> result_finetune.txt
