import os
import glob
import json
import numpy as np
import pandas as pd

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "mountaincar*.csv")


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

        # =========================
        # FIX SUCCESS COLUMN
        # =========================
        #
        # MountainCar solves when:
        # episode length < 200
        #
        # failed runs usually hit max episode length
        #

        episode_lengths = (
            df.groupby(["user_id", "episode"])["step"]
            .transform("count")
        )

        df["success"] = episode_lengths < 200

        # overwrite CSV safely
        tmp_path = f + ".tmp"
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, f)

        dfs.append(df)

        print(f"Loaded {f} with {len(df)} rows")

    if not dfs:
        raise RuntimeError("No MountainCar data found.")

    df = pd.concat(dfs, ignore_index=True)

    print("\nTotal rows:", len(df))

    # =========================
    # PARSE STATES
    # =========================

    df["state_parsed"] = df["state"].apply(parse_state)

    before = len(df)

    df = df.dropna(subset=["state_parsed"])

    after = len(df)

    print(f"Dropped {before - after} bad rows")

    # =========================
    # REMOVE TRAINING MODE
    # =========================

    if "training" in df.columns:
        df = df[df["training"] != True]

    print("Rows after removing training data:", len(df))

    # =========================
    # REMOVE UNFINISHED EPISODES
    # =========================

    completed = (
        df.groupby(["user_id", "episode"])["done"]
        .any()
        .reset_index()
    )

    completed = completed[completed["done"] == True]

    df = df.merge(
        completed[["user_id", "episode"]],
        on=["user_id", "episode"]
    )

    print("Rows after removing unfinished episodes:", len(df))

    # =========================
    # BUILD DATASET
    # =========================

    X = np.vstack(df["state_parsed"].values)
    y = df["action"].astype(int).values

    print("\nDataset shape:", X.shape)
    print("Action distribution:", np.bincount(y))

    return df, X, y
