# AmeircasNLP2023

We will be using Meta's No Language Left Behind Model (NLLB) for the machine translation tasks

## Results:
| **Baseline NLLB-600M**   | **chrF** | **BLEU** |
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


| **Finetuned NLLB-600M**  | **chrF**.| **BLEU** |
| -------------------------| -------- | -------- |
| `1. ashaninka -> es`     | 17.11    | 1.87     |
| `2. aymara -> es`        | 33.83    | 12.85    |
| `3. bribri -> es`        | 27.05    | 5.90     |
| `4. guarani -> es`       | 34.43    | 12.93    |
| `5. hñähñu -> es`        | 17.09    | 1.27     |
| `6. nahuatl -> es`       | 19.64    | 3.41     |
| `7. quechua -> es`       | 38.12    | 14.33    |
| `8. raramuri -> es`      | 20.58    | 2.27     |
| `9. shipibo_konibo -> es`| 36.10    | 12.96    |
| `10. wixarika -> es`     | 23.84    | 4.70     |


| **Baseline NLLB-1.3B**   | **chrF**.| **BLEU** |
| -------------------------| -------- | -------- |
| `1. ashaninka -> es`     | `15.13`  | 1.60     |
| `2. aymara -> es`        | `21.64`  | `4.78`   |
| `3. bribri -> es`        |  17.53   | 1.59     |
| `4. guarani -> es`       | `21.43`  | `5.54`   |
| `5. hñähñu -> es`        | `16.84`  | `1.45`   |
| `6. nahuatl -> es`       | `14.57`  | `1.08`   |
| `7. quechua -> es`       | `28.20`  | `9.36`   |
| `8. raramuri -> es`      | `15.35`  | `1.49`   |
| `9. shipibo_konibo -> es`| `20.52`  | `2.88`   |
| `10. wixarika -> es`     | `14.62`  | 1.02     | 

![alt text](https://github.com/KaieChen/ameircasnlp2023/blob/main/output.png)
