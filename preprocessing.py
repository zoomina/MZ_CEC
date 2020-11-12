from torch.utils.data import Dataset
from kobert.utils import get_tokenizer

import pandas as pd
import re
from sklearn.model_selection import train_test_split

from model.kobert import *

def preprocessing():
    trainpath = './data/train_df.csv'
    devpath = './data/dev_df.csv'
    train = pd.read_csv(trainpath)
    dev = pd.read_csv(devpath)

    dev['emotion_u'] -= 1
    train['emotion_u'] -= 1

    p = re.compile("[{}:H'S0123,]")
    train['content'] = list(map(lambda sent: str((p.sub('', sent))), train['content']))
    dev['content'] = list(map(lambda sent: str((p.sub('', sent))), dev['content']))

    train_df = train[['content', 'emotion_u']]
    dev_df = dev[['content', 'emotion_u']]

    dtls = [list(train_df.iloc[i, :]) for i in range(len(train_df))]
    eval_dtls = [list(dev_df.iloc[i, :]) for i in range(len(dev_df))]
    return dtls, eval_dtls

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

