import pandas as pd
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm


class ElectraDataset(Dataset):
    def __init__(self, csv_file):
        # 일부 값중에 NaN이 있음...
        self.dataset = pd.read_csv(csv_file, sep='\t').dropna(axis=0)
        # 중복제거
        self.dataset.drop_duplicates(subset=['document'], inplace=True)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:3].values
        text = row[0]
        y = row[1]

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256,
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0]
        attention_mask = inputs['attention_mask'][0]

        return input_ids, attention_mask, y

## 최종 output_dim==2라서 추가적인 tuning이 필요할 듯 ##
## 그래서 당장 사용할 수 있을지 모르겠음 ##
class ElectraClassifier(nn.Module):
    def __init__(self, input_ids_batch, attention_mask, hidden_size=768, num_classes=6, dr_rate=None, params=None):
        super(ElectraClassifier, self).__init__()
        self.dr_rate = dr_rate
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.electramodel = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v2-discriminator")
        self.attention_mask = attention_mask
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:

            self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, segment_ids):
        attention_mask = self.attention_mask

        _, pooler = self.electramodel(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

    def build_model(self):
        model = ElectraClassifier(dr_rate=0.5).to(self.device)
        return model.to(self.device)
