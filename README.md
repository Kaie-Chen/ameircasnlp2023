# AmeircasNLP2023

We will be using Meta's No Language Left Behind Model (NLLB) for the machine translation tasks

## Result:

| **X-to-Spanish **| **Baseline (Test)** | **Multi** | **Multi+** | **Multi++ **| **Bi** | **Bi++** |** Bi++ (Test)**|
| -------------|------------------|-------|--------|---------|----|-------|-----------|
|Wixarika| 0.304 |0.277 |0.294 |0.294 |0.266 |0.279| 0.288|
|Hñähñu |0.147| 0.129| 0.133 |0.138 |0.144 |0.141 |0.148|
|Aymara |0.283 |0.291| 0.328 |0.326 |0.336 |0.326 |0.300|
|Shipibo-Konibo| 0.329 |0.224 |0.238 |0.253 |0.261 |0.283 |0.277|
|Nahuatl |0.266| 0.241 |0.252 |0.275| 0.282 |0.283 |0.237|
|Guarani |0.336 |0.304 |0.316 |0.321| 0.315| 0.303 |0.331|
|Asháninka |0.258 |0.222 |0.238 |0.272 |0.269 |0.286| 0.280|
|Quechua |0.343| 0.324 |0.341|  |0.337 | |0.344|
|Rarámuri |0.184| 0.161| 0.175|  |0.184| | 0.145|
|Bribri |0.165| 0.210| 0.237 ||0.231| |0.148|
Table 1: Result in ChrF++ on develop dataset, except for baseline and Bi++(test). Baseline model is the best submission for AmericasNLP 2021. The effectiveness of weight averaging (Multi+ and Bi+) and back translation is compared (Multi++ and Bi++). We also compared the performance of bilingual (Bi) and multilingual (Multi).
