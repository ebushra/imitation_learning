import os
import glob
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

import gymnasium as gym
from sklearn.neighbors import NearestNeighbors
import numpy as np

# =========================
# CONFIG
# =========================

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "acrobot*.csv")

# =========================
# PARSE STATE
# =========================

def parse_state(s):
    try:
        if not isinstance(s, str):
            return None

        return np.array(json.loads(s), dtype=float)

    except:
        return None

# =========================
# MODEL ROLLOUT
# =========================

def rollout_model(model, scaler, episodes=5):

    env = gym.make("Acrobot-v1")

    rollout_lengths = []

    for ep in range(episodes):

        obs, _ = env.reset()

        done = False
        steps = 0

        while not done and steps < 500:

            state = np.array(obs, dtype=float).reshape(1, -1)

            state = scaler.transform(state)

            action = model.predict(state)[0]

            obs, reward, terminated, truncated, _ = env.step(int(action))

            done = terminated or truncated

            steps += 1

        rollout_lengths.append(steps)

    env.close()

    return rollout_lengths

# =========================
# MAIN
# =========================

files = glob.glob(PATTERN)

print("\nFound files:")
for f in files:
    print(" -", f)

for f in files:

    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(f))
    print("=" * 70)

    try:

        # =========================
        # LOAD CSV
        # =========================

        df = pd.read_csv(f)

        print("Rows:", len(df))

        # remove training rows ONLY
        if "training" in df.columns:
            df = df[df["training"] != True]

        # parse states
        df["state_parsed"] = df["state"].apply(parse_state)

        before = len(df)

        df = df.dropna(subset=["state_parsed"])

        print("Dropped bad rows:", before - len(df))

        # =========================
        # HUMAN EPISODE LENGTHS
        # =========================

        human_episode_lengths = (
            df.groupby(["user_id", "episode"])["step"]
            .max()
            .values
        )

        avg_human_length = np.mean(human_episode_lengths)

        print("\nAverage human episode length:")
        print(avg_human_length)

        # =========================
        # BUILD DATASET
        # =========================

        X = np.vstack(df["state_parsed"].values)
        y = df["action"].astype(int).values

        print("\nDataset shape:", X.shape)

        # skip tiny datasets
        if len(X) < 50:
            print("Skipping: not enough data")
            continue

        # =========================
        # TRAIN / TEST SPLIT
        # =========================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # =========================
        # MODEL
        # =========================

        model = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            max_iter=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        # =========================
        # ACCURACY
        # =========================

        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print("\nModel accuracy:")
        print(accuracy)

        # =========================
        # ROLLOUT
        # =========================

        rollout_lengths = rollout_model(model, scaler)

        print("\nRollout episode lengths:")
        print(rollout_lengths)

        print("\nAverage rollout length:")
        print(np.mean(rollout_lengths))

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)
        
        distances, _ = nn.kneighbors(X_rollout)
        
        print("Mean distance to training set:", distances.mean())
        print("Median distance:", np.median(distances))
        print("Max distance:", distances.max())
        overlap = np.exp(-distances.mean())
        print("Overlap: ", overlap)

    except Exception as e:

        print("\nFAILED:")
        print(e)
