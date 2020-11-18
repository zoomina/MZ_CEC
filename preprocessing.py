from torch.utils.data import Dataset
from kobert.utils import get_tokenizer

import pandas as pd
import re
from sklearn.model_selection import train_test_split

from model.kobert import *

def json2csv(data, test=False):
    df = pd.DataFrame(columns=["emotion_u", "emotion_i", "age", "gender", "situation", "disease", "content"])

    with open(data, "r", encoding="utf-8") as file:
        root = next(file).strip()
        for line in file:
            line = eval(line[:-2])  # keys = profile, talk
            profile_info = line["profile"]  # keys = persona-id, persona, emotion
            talk_info = line["talk"]  # keys = id, content
            if test:
                inner = {"talk_id": talk_info["id"]["talk-id"], "emotion_u": 99, "emotion_i": 99,
                         "age": profile_info["persona"]["human"][0], "gender": profile_info["persona"]["human"][1],
                         "situation": profile_info["emotion"]["situation"][0],
                         "disease": profile_info["emotion"]["situation"][0],
                         "content": talk_info["content"]}
            else:
                inner = {"talk_id": talk_info["id"]["talk-id"], "emotion_u": profile_info["emotion"]["type"][1],
                         "emotion_i": profile_info["emotion"]["type"],
                         "age": profile_info["persona"]["human"][0], "gender": profile_info["persona"]["human"][1],
                         "situation": profile_info["emotion"]["situation"][0],
                         "disease": profile_info["emotion"]["situation"][0],
                         "content": talk_info["content"]}
            df = df.append(inner, ignore_index=True)

    df['content'] = list(map(lambda dic: ' '.join(dic.values()), df['content']))

    return df

def preprocessing(df, inner_emotion=-1, test=False):
    if test:
        if inner_emotion == -1:
            df_new = df[['content', 'emotion_u']]
        else:
            df_new = df[['content', 'emotion_upper']][df['emotion_upper'] == inner_emotion]

        dtls = [list(df_new.iloc[i, :]) for i in range(len(df_new))]
        return dtls

    df['emotion_u'] = df['emotion_u'].apply(lambda x : int(x) - 1)  #astype(int)   #

    if inner_emotion == -1:
        df_new = df[['content', 'emotion_u']]
    else:
        df_new = df[['content', 'emotion_upper']][df['emotion_upper'] == inner_emotion]

    dtls = [list(df_new.iloc[i, :]) for i in range(len(df_new))]

    return dtls

def data_loader(dtls, max_len, batch_size, num_workers):
    dataset_train, dataset_test = train_test_split(dtls, test_size=0.2, random_state=123)

    tokenizer = get_tokenizer()
    _, vocab = get_pytorch_kobert_model()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
    data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, test_dataloader

def test_loader(dtls, max_len, batch_size, num_workers):
    tokenizer = get_tokenizer()
    _, vocab = get_pytorch_kobert_model()
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    data_test = BERTDataset(dtls, 0, 1, tok, max_len, True, False)

    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)

    return test_dataloader

