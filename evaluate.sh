echo "---------ashaninka---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/ashaninka/SpanishFromAshaninka.txt  --gold_reference ./processed_data/ashaninka/dev.es >> result.txt
echo "---------aymara---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/aymara/SpanishFromAymara.txt  --gold_reference ./processed_data/aymara/dev.es >> result.txt
echo "---------bribri---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/bribri/SpanishFromBribri.txt  --gold_reference ./processed_data/bribri/dev.es >> result.txt
echo "---------guarani---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/guarani/SpanishFromGuarani.txt  --gold_reference ./processed_data/guarani/dev.es >> result.txt
echo "---------hñähñu---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/hñähñu/SpanishFromhNähñu.txt  --gold_reference ./processed_data/hñähñu/dev.es >> result.txt
echo "---------nahuatl---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/nahuatl/SpanishFromNahuatl.txt  --gold_reference ./processed_data/nahuatl/dev.es >> result.txt
echo "---------quechua---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/quechua/SpanishFromQuechua.txt  --gold_reference ./processed_data/quechua/dev.es >> result.txt
echo "---------raramuri---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/raramuri/SpanishFromRaramuri.txt  --gold_reference ./processed_data/raramuri/dev.es >> result.txt
echo "---------shipibo_konibo---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/shipibo_konibo/SpanishFromShipibo_konibo.txt  --gold_reference ./processed_data/shipibo_konibo/dev.es >> result.txt
echo "---------wixarika---------" >> result.txt
python3 evaluate.py --system_output ./processed_data/wixarika/SpanishFromWixarika.txt  --gold_reference ./processed_data/wixarika/dev.es >> result.txt
