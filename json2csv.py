import argparse
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.DataFrame(columns=["emotion_u", "emotion_i", "age", "gender", "situation", "disease", "content"])

    with open(args.data, "r", encoding="utf-8") as file:
        root = next(file).strip()
        for line in file:
            line = eval(line[:-2])  # keys = profile, talk
            profile_info = line["profile"]  # keys = persona-id, persona, emotion
            talk_info = line["talk"]  # keys = id, content
            inner = {"emotion_u": profile_info["emotion"]["type"][1], "emotion_i": profile_info["emotion"]["type"],
                     "age" : profile_info["persona"]["human"][0], "gender" : profile_info["persona"]["human"][1],
                     "situation" : profile_info["emotion"]["situation"][0], "disease" : profile_info["emotion"]["situation"][0],
                     "content": talk_info["content"]}
            df = df.append(inner, ignore_index=True)

    df.to_csv(os.path.join(args.output_dir, args.output_name + ".csv"))

