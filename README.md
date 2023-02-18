# ESGCN

## Datasets

> Please first download the datasets [here](https://drive.google.com/drive/folders/1SN3JAV3clMMUPQ0M6LTJQ4GZ8JFLTy0s?usp=sharing) and extract them into `data/` directory.

Initial datasets DBP15K and DWY100K are from [JAPE](https://github.com/nju-websoft/JAPE) and [BootEA](https://github.com/nju-websoft/BootEA).

Take the dataset DBP15K (ZH-EN) as an example, the folder "zh_en" contains:
* ent_ids_1: ids for entities in source KG;
* ent_ids_2: ids for entities in target KG;
* ref_ent_ids: entity links encoded by ids;
* triples_1: relation triples encoded by ids in source KG (DBP_ZH);
* triples_2: relation triples encoded by ids in target KG (DBP_EN);

## Environment

* Python>=3.5
* Tensorflow>=1.8.0
* Scipy
* Numpy

> Due to the limited graphics memory of GPU, we ran our codes using CPUs (40  Intel(R) Xeon(R) CPU E5-2640 v4 @ 2.40GHz).

## Running

For example, to run NMN on DBP15K (ZH-EN), use the following script:
```
python3 main.py
```

> Due to the instability of embedding-based methods, it is acceptable that the results fluctuate a little bit (±1%) when running code repeatedly.

