# MZ_CEC

<br>

MZ CEC 감정인식대회 해커톤 emo:)tale 팀 : 박건우, 변자민  
  
장려상 수상 (4등)

MZ CEC Classification Emotional Conversation Hackathon Team emo:)tale : Geonu Park, Jamin Byeon  
4th  

<br>

## Requirement

```
mxnet==1.6.0
gluonnlp==0.9.0
pandas
tqdm
sentencepiece==0.1.85
transformers==2.1.1
torch  #원래 ==1.3.1
numpy<1.19.4

#SKT에서 공개한 KoBERT 모델을 불러옵니다 
!pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
```

<br>

## Data Structure

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

> **result**
>
> ```
> pd.DataFrame({"emotion_u": profile_info["emotion"]["type"][1], "emotion_i": profile_info["emotion"]["type"],
>               "age" : profile_info["persona"]["human"][0], "gender" : profile_info["persona"]["human"][1],
>               "situation" : profile_info["emotion"]["situation"][0], "disease" : profile_info["emotion"]["situation"][0],
>               "content": talk_info["content"]})
> ```

<br>

----

<br>

## Model

SKTBrain KoBERT를 finetuning하여 사용  
  
  

![model](https://user-images.githubusercontent.com/39390943/100821973-c6cc1600-3494-11eb-84ef-286562a9b3de.png)

<br>

### train

```
python ./train.py --batch-size <batch_size> --num-epochs <num_epochs> --num-workers <num_workers> --num-classes <num_classes::int>
```

### test

```
python ./test.py <checkpoint_path> --batch-size <batch_size> --num-workers <num_workers> --num-classes <num_classes::int> --small-emotion <small_emotion::int>
```

### evaluation

```
python ./evaluation.py <data_path>
```
<br>

## Result

감정 대분류(6개 감정) 정확도 : 약 0.61  
Accuracy for classification on 6 emotion : about 0.61  
감정 소분류(각 10개 감정) 정확도 :    
Accuracy for classification on 60 emotion (10 on each category above) :
  U1: 0.46   
  U2: 0.49   
  U3: 0.59   
  U4: 0.62   
  U5: 0.59   
  U6: 0.46   
최종(60개 감정) 정확도 : 0.33  
Final accuracy on 60 emotion : 0.33
