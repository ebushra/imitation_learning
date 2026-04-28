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
# ROLLOUT
# =========================

def rollout_model(model, scaler, episodes=25):

    env = gym.make("Acrobot-v1")

    rollout_lengths = []
    all_states = []
    successes = 0

    for ep in range(episodes):

        obs, _ = env.reset()
        done = False
        steps = 0
        episode_states = []

        while not done and steps < 500:

            state = np.array(obs, dtype=float).reshape(1, -1)
            state = scaler.transform(state)

            action = model.predict(state)[0]

            obs, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            episode_states.append(obs)
            steps += 1

        rollout_lengths.append(steps)
        all_states.extend(episode_states)

        if steps < 500:
            successes += 1

    env.close()

    success_rate = successes / episodes

    return rollout_lengths, np.array(all_states), success_rate

# =========================
# MAIN
# =========================

files = glob.glob(PATTERN)

results = {}

print("\nFound files:")
for f in files:
    print(" -", f)

for f in files:

    print("\n" + "=" * 70)
    print("FILE:", os.path.basename(f))
    print("=" * 70)

    try:

        df = pd.read_csv(f)

        print("Rows:", len(df))

        if "training" in df.columns:
            df = df[df["training"] != True]

        df["state_parsed"] = df["state"].apply(parse_state)

        before = len(df)
        df = df.dropna(subset=["state_parsed"])
        print("Dropped bad rows:", before - len(df))

        # =========================
        # HUMAN STATS
        # =========================

        human_episode_lengths = (
            df.groupby(["user_id", "episode"])["step"]
            .max()
            .values
        )

        avg_human_length = np.mean(human_episode_lengths)

        print("\nAverage human episode length:", avg_human_length)

        # =========================
        # DATASET
        # =========================

        X = np.vstack(df["state_parsed"].values)
        y = df["action"].astype(int).values

        if len(X) < 50:
            print("Skipping: not enough data")
            continue

        # =========================
        # TRAIN / TEST
        # =========================

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = MLPClassifier(
            hidden_layer_sizes=(64, 64),
            max_iter=200,
            random_state=42
        )

        model.fit(X_train, y_train)

        accuracy = accuracy_score(y_test, model.predict(X_test))

        print("\nModel accuracy:", accuracy)

        # =========================
        # ROLLOUT
        # =========================

        rollout_lengths, X_rollout, rollout_success = rollout_model(model, scaler)

        avg_rollout_length = np.mean(rollout_lengths)

        print("\nRollout lengths:", rollout_lengths)
        print("Average rollout length:", avg_rollout_length)
        print("Rollout success rate:", rollout_success)

        # =========================
        # OVERLAP
        # =========================

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_train)

        distances, _ = nn.kneighbors(X_rollout)

        overlap = np.exp(-distances.mean())

        print("Overlap:", overlap)

        # =========================
        # STORE RESULTS
        # =========================

        results[os.path.basename(f)] = {
            "human_episode_len": float(avg_human_length),
            "accuracy": float(accuracy),
            "rollout_len": float(avg_rollout_length),
            "overlap": float(overlap),
            "rollout_success": float(rollout_success),
        }

    except Exception as e:
        print("\nFAILED:")
        print(e)

# =========================
# FINAL SUMMARY
# =========================

print("\n\n================ FINAL RESULTS ================\n")
print(json.dumps(results, indent=4))
