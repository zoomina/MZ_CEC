import argparse
import os
import json
from torch.utils.data import Dataset
from kobert.utils import get_tokenizer
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from model.kobert import *
import pandas as pd
import argparse
from tqdm import tqdm
from tqdm.notebook import tqdm

def json2csv(data):
  df = pd.DataFrame(columns=["talk_id","emotion_u", "emotion_i", "age", "gender", "situation", "disease", "content"])

  with open(data, "r", encoding="utf-8") as file:
    root = next(file).strip()
    for line in file:
      line = eval(line[:-2])  # keys = profile, talk
      profile_info = line["profile"]  # keys = persona-id, persona, emotion
      talk_info = line["talk"]  # keys = id, content
      inner = {"talk_id": talk_info["id"]["talk-id"], "emotion_u": 99, "emotion_i": 99,
                "age" : profile_info["persona"]["human"][0], "gender" : profile_info["persona"]["human"][1],
                "situation" : profile_info["emotion"]["situation"][0], "disease" : profile_info["emotion"]["situation"][0],
                "content": talk_info["content"]}
      df = df.append(inner, ignore_index=True)

  df['content'] = list(map(lambda dic: ' '.join(dic.values()), df['content']))

  return df

def preprocessing(df, inner_emotion=-1):
  if inner_emotion == -1:
    df_new = df[['content', 'emotion_u']]
  else:
    df_new = df[['content','emotion_upper']][df['emotion_upper']==inner_emotion]

  dtls = [list(df_new.iloc[i, :]) for i in range(len(df_new))]
  return dtls

def test_loader(dtls, max_len, batch_size, num_workers):
  tokenizer = get_tokenizer()
  _, vocab = get_pytorch_kobert_model()
  tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

  data_test = BERTDataset(dtls, 0, 1, tok, max_len, True, False)

  test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=num_workers)

  return test_dataloader

def test(test_dataloader, model, device):
  model.eval()
  answer=[]
  test_acc = 0.0
  with torch.no_grad():
      for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
          token_ids = token_ids.long().to(device)
          segment_ids = segment_ids.long().to(device)
          valid_length= valid_length
          out = model(token_ids, valid_length, segment_ids)
          max_vals, max_indices = torch.max(out, 1)
          answer.extend(max_indices.cpu().clone().numpy())
  preds = np.array(answer).flatten()
  result = pd.DataFrame({"pred": preds})
  return result


def test_main(df, path, batch_size, num_workers, inner_emotion, num_classes=6):
  '''
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint")
  parser.add_argument("--batch-size", type=int, default=64, help="default=64")
  parser.add_argument("--num-workers", type=int, default=5, help="default=5")
  parser.add_argument("--num-classes", type=int, default=6, help="default=5")
  parser.add_argument("--small-emotion", type=int, default=-1, help="default=-1")
  args = parser.parse_args()
  
  batch_size = args.batch_size
  num_workers = args.num_workers
  small_emotion = args.small_emotion
  checkpoint = torch.load(args.checkpoint) #"result/epoch3_batch24.pt"
  '''
  checkpoint = torch.load(path)

  max_len = 64
  model = BERTClassifier(num_classes=6).build_model()
  model.load_state_dict(checkpoint)
  device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  eval_dtls = preprocessing(df, inner_emotion)

  test_dataloader = test_loader(eval_dtls, max_len, batch_size, num_workers)
  
  result_df = test(test_dataloader, model, device)
  final = pd.concat([df['content'], result_df], ignore_index=True, axis=1)
  final.columns=['content','pred']
  return final


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("path")
  args = parser.parse_args()
  
  path_U = './model/trainU_epoch4_batch64.pt'
  df = json2csv(args.path)
  emotion_U = test_main(df, path_U, batch_size=64, num_workers=5, inner_emotion=-1)
  emotion_U.columns=['content','emotion_upper']
  emotion_I = pd.DataFrame(columns=['content','pred'])

  for inner_emotion in range(0,6): ######changed
    path_I = './model/trainU{}_epoch4_batch32.pt'.format(inner_emotion+1)
    # temp_emo = test_main(emotion_U, path_I, batch_size=64, num_workers=5, inner_emotion=inner_emotion, num_classes=10)
    checkpoint = torch.load(path_I)
    max_len = 64
    model2 = BERTClassifier(num_classes=10).build_model()
    model2.load_state_dict(checkpoint)
    df_new = emotion_U[['content','emotion_upper']][emotion_U['emotion_upper']==inner_emotion]
    eval_dtls = preprocessing(df_new, inner_emotion)

    test_dataloader = test_loader(eval_dtls, max_len, batch_size=64, num_workers=5)

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    result_df = test(test_dataloader, model2, device)
    final = pd.concat([df_new['content'], result_df], ignore_index=True, axis=1)
    final.columns=['content','pred']
    temp_emo = pd.merge(final, df_new[['content','emotion_upper']], how='inner', on = 'content')
    temp_emo = temp_emo.dropna()
    emotion_I=emotion_I.append(temp_emo)
  
  final = pd.merge(emotion_I, df[['talk_id','content']], how='inner', on='content')
  
  fin_dict = {final["talk_id"][i]:"E"+str(int(final["emotion_upper"][i]+1))+str(int(final["pred"][i])) for i in range(len(final))}

  if not os.path.exists("./result"):
    os.mkdir("./result")
  
  with open('./result/prediction.json','w') as f:
    json.dump(fin_dict,f)

# !python evaluation.py <data-path>








