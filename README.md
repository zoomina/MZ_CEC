# MZ_CEC

<br>

MZ CEC 감정인식대회 해커톤 emo:)tale 팀 : 박건우, 변자민

<br>

## Requirement

```
!pip install mxnet==1.6.0
!pip install gluonnlp==0.9.0
!pip install pandas tqdm
!pip install sentencepiece==0.1.85
!pip install transformers==2.1.1
!pip install torch  #원래 ==1.3.1

#SKT에서 공개한 KoBERT 모델을 불러옵니다 
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```

<br>

## Data Sample

```
[
 {
   "profile": {
      "profile-id": "Pro_091",
      "persona": {
         "persona-id": "A01_G01_C01",
         "human": [
            "A01",
            "G01"
         ],
         "computer": [
            "C01"
         ]
      },
      "emotion": {
         "emotion-id": "S01_D02_E31",
         "type": "E31",
         "situation": [
            "S01",
            "D02"
         ]
      }
   },
   "talk": {
      "id": {
         "profile-id": "Pro_091",
         "talk-id": "Pro_091_01226"
      },
      "content": {
         "HS01": "-대화-",
         "SS01": "-대화-",
         "HS02": "-대화-",
         "SS02": "-대화-",
         "HS03": "-대화-",
         "SS03": "-대화-"
      }
}
]
```

### Preprocessing

- json2csv.py
```
python ./json2csv.py <input_json_data> --output-dir <output_dir> --output-name <output_file_name_without_extension>
```

<br>

----

<br>

## Model


### train

```
python ./train.py --batch-size <batch_size> --num-epochs <num_epochs> --num-workers <num_workers>
```

### test


<br>

## Result
