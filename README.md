# AmeircasNLP2023

We will be using Meta's No Language Left Behind Model (NLLB) for the machine translation tasks

## Results:
|               --         |**Baseline NLLB-600M**|
| -------------------------| -------- | -------- |
|                  --      | **chrF** | **BLEU** |
| -------------------------| -------- | -------- |
| `1. ashaninka -> es`     | 16.56    | 2.36     |
| `2. aymara -> es`        | 30.32    | 9.82     |
| `3. bribri -> es`        | 21.37    | 2.45     |
| `4. guarani -> es`       | 35.58    | 14.35    |
| `5. hñähñu -> es`        | 17.53    | 1.30     |
| `6. nahuatl -> es`       | 12.98    | 0.92     |
| `7. quechua -> es`       | 32.56    | 10.35    |
| `8. raramuri -> es`      | 17.07    | 1.19     |
| `9. shipibo_konibo -> es`| 22.57    | 3.24     |
| `10. wixarika -> es`     | 16.61    | 1.61     |


| **Finetuned NLLB-600M**  | **chrF** | **BLEU** |
| -------------------------| -------- | -------- |
| `1. ashaninka -> es`     | 20.47    | 3.35     |
| `2. aymara -> es`        | 32.03    | 11.03    |
| `3. bribri -> es`        | 31.41    | 8.60     |
| `4. guarani -> es`       | 30.98    | 9.79     |
| `5. hñähñu -> es`        | 16.33    | 1.13     |
| `6. nahuatl -> es`       | 27.21    | 8.60     |
| `7. quechua -> es`       | 34.64    | 10.83    |
| `8. raramuri -> es`      | 23.51    | 3.68     |
| `9. shipibo_konibo -> es`| 37.01    | 13.46    |
| `10. wixarika -> es`     | 27.68    | 6.36     |

| **Averaged NLLB-600M**   | **chrF** | **BLEU** |
| -------------------------| -------- | -------- |
| `1. ashaninka -> es`     | 21.26    | 3.80     |
| `2. aymara -> es`        | 34.48    | 13.23    |
| `3. bribri -> es`        | 32.79    | 10.03    |
| `4. guarani -> es`       | 34.15    | 13.01    |
| `5. hñähñu -> es`        | 17.54    | 1.45     |
| `6. nahuatl -> es`       | 27.78    | 9.01     |
| `7. quechua -> es`       | 37.61    | 14.23    |
| `8. raramuri -> es`      | 24.43    | 4.33     |
| `9. shipibo_konibo -> es`| 39.60    | 16.03    |
| `10. wixarika -> es`     | 28.30    | 6.91     |


Multi-lingual model log
![alt text](https://github.com/KaieChen/ameircasnlp2023/blob/main/output.png)
