import os
import json

from test import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path of json data")
    args = parser.parse_args()

    path_U = './result/trainU_epoch4_batch32.pt'
    df = json2csv(args.path)
    emotion_U = test_main(path_U, args.path, batch_size=64, num_workers=5, inner_emotion=-1)
    emotion_U.columns = ['content', 'emotion_upper']
    emotion_I = pd.DataFrame(columns=['content', 'pred'])

    for inner_emotion in range(0, 6):
        path_I = './result/trainU{}_epoch4_batch32.pt'.format(inner_emotion + 1)
        temp_emo = test_main(emotion_U, path_I, batch_size=64, num_workers=5, inner_emotion=inner_emotion, num_classes=10, test=True)
        temp_emo = temp_emo.dropna()
        emotion_I = emotion_I.append(temp_emo)

    final = pd.merge(emotion_I, df[['talk_id', 'content']], how='inner', on='content')

    fin_dict = {final["talk_id"][i]: "E" + str(int(final["emotion_upper"][i] + 1)) + str(int(final["pred"][i])) for i in
                range(len(final))}

    if not os.path.exists("./result"):
        os.mkdir("./result")

    with open('./result/prediction.json', 'w') as f:
        json.dump(fin_dict, f)