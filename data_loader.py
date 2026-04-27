import os
import glob
import json
import numpy as np
import pandas as pd

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "acrobot*.csv")


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
        print(" -", f)
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    df["state_parsed"] = df["state"].apply(parse_state)
    df = df.dropna(subset=["state_parsed"])

    X = np.vstack(df["state_parsed"].values)
    y = df["action"].astype(int).values

    print("\nDataset shape:", X.shape)
    print("Action distribution:", np.bincount(y))

    return df, X, y
