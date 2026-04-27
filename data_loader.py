import os
import glob
import json
import numpy as np
import pandas as pd

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "acrobot_ba0607dd-1bbe-4ef0-a267-0eae6365ba20.csv")


def parse_state(s):
    try:
        if not isinstance(s, str):
            return None
        return np.array(json.loads(s), dtype=float)
    except:
        return None


def load_data():
    files = glob.glob(PATTERN)

    print("\nFound files:")
    dfs = []

    for f in files:
        df = pd.read_csv(f)

        df["source_file"] = os.path.basename(f)
    
        episode_lengths = df.groupby(["user_id", "episode"])["step"].transform("count")
        df["success"] = episode_lengths < 500  # ONLY column changed
    
        tmp_path = f + ".tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, f)  # atomic swap
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["state_parsed"] = df["state"].apply(parse_state)
    df = df.dropna(subset=["state_parsed"])

    X = np.vstack(df["state_parsed"].values)
    y = df["action"].astype(int).values

    print("\nDataset shape:", X.shape)
    print("Action distribution:", np.bincount(y))

    return df, X, y
