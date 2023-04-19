echo "ashaninka: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/ashaninka/SpanishFromAshaninka.txt  --gold_reference ./processed_data/ashaninka/dev.es > evaluation.txt
echo "\naymara: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/aymara/SpanishFromAymara.txt  --gold_reference ./processed_data/aymara/dev.es > evaluation.txt
echo "\nbribri: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/bribri/SpanishFromBribri.txt  --gold_reference ./processed_data/bribri/dev.es > evaluation.txt
echo "\nguarani: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/guarani/SpanishFromGuarani.txt  --gold_reference ./processed_data/guarani/dev.es > evaluation.txt
echo "\nhñähñu: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/hñähñu/SpanishFromhNähñu.txt  --gold_reference ./processed_data/hñähñu/dev.es > evaluation.txt
echo "\nnahuatl: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/nahuatl/SpanishFromNahuatl.txt  --gold_reference ./processed_data/nahuatl/dev.es > evaluation.txt
echo "\nquechua: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/quechua/SpanishFromQuechua.txt  --gold_reference ./processed_data/quechua/dev.es > evaluation.txt
echo "\nraramuri: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/raramuri/SpanishFromRaramuri.txt  --gold_reference ./processed_data/raramuri/dev.es > evaluation.txt
echo "\nshipibo_konibo: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/shipibo_konibo/SpanishFromShipibo_konibo.txt  --gold_reference ./processed_data/shipibo_konibo/dev.es > evaluation.txt
echo "\nwixarika: \n" | tee evaluation.txt
python3 evaluate.py --system_output ./processed_data/wixarika/SpanishFromWixarika.txt  --gold_reference ./processed_data/wixarika/dev.es > evaluation.txt


