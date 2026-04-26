import os
import glob
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# =========================
# CONFIG
# =========================

DATA_DIR = "/var/data/human_data"
PATTERN = os.path.join(DATA_DIR, "acrobot*.csv")

# =========================
# LOAD FILES
# =========================

files = glob.glob(PATTERN)

print("\nFound files:")
for f in files:
    print(" -", f)

dfs = []

for f in files:
    try:
        df_part = pd.read_csv(f)
        df_part["source_file"] = os.path.basename(f)
        dfs.append(df_part)
        print(f"Loaded {f} with {len(df_part)} rows")
    except Exception as e:
        print(f"Failed to load {f}: {e}")

if not dfs:
    raise RuntimeError("No data loaded.")

df = pd.concat(dfs, ignore_index=True)

print("\nTotal rows:", len(df))

# =========================
# PARSE STATE (FIXED ORDER)
# =========================

def parse_state(s):
    try:
        if not isinstance(s, str):
            return None
        return np.array(json.loads(s), dtype=float)
    except:
        return None


df["state_parsed"] = df["state"].apply(parse_state)

print("NaN states:", df["state_parsed"].isna().sum())
print("Example raw:", df["state"].iloc[0])
print("Example parsed:", df["state_parsed"].iloc[0])

# =========================
# CLEAN DATA
# =========================

before = len(df)
df = df.dropna(subset=["state_parsed"])
after = len(df)

print(f"Dropped {before - after} bad rows")

# =========================
# BUILD DATASET
# =========================

X = np.vstack(df["state_parsed"].values)
y = df["action"].astype(int).values

print("\nDataset shape:", X.shape)
print("Actions distribution:", np.bincount(y))

if len(X) == 0:
    raise RuntimeError("No valid state data after parsing.")

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# MODEL
# =========================

model = MLPClassifier(
    hidden_layer_sizes=(64, 64),
    max_iter=100,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# EVALUATION
# =========================

y_pred = model.predict(X_test)

print("\n=== TEST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix on test set:")
print("Rows = TRUE, Cols = PRED")
print("      0   1   2")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\n=== TRAIN RESULTS ===")
print("Accuracy:", accuracy_score(y_train, y_pred_train))

print("\nConfusion Matrix (Train):")
print("Rows = TRUE, Cols = PRED")
print("      0   1   2")
print(confusion_matrix(y_train, y_pred_train))

# =========================
# BASELINES
# =========================

random_preds = np.random.choice(np.unique(y), size=len(y_test))
majority_class = np.bincount(y_train).argmax()
majority_preds = np.full_like(y_test, majority_class)

print("\n=== BASELINES ===")
print("Random:", accuracy_score(y_test, random_preds))
print("Majority:", accuracy_score(y_test, majority_preds))

# =========================
# EXTRA INSIGHT
# =========================

print("\nAverage human episode length (steps):")
print(df.groupby("episode")["step"].max().mean())

print("\nSuccess rate:")
print(df["done"].mean())
    
# =========================
# MODEL ROLLOUT EVALUATION
# =========================

import gymnasium as gym

print("\n=== MODEL ROLLOUT (5 EPISODES) ===")

env = gym.make("Acrobot-v1")

def obs_to_model_state(obs):
    return np.array(obs, dtype=float)

num_episodes = 5
episode_steps = []
successes = 0

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 500:  # safety cap
        state = obs_to_model_state(obs).reshape(1, -1)

        action = model.predict(state)[0]

        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

        steps += 1

    episode_steps.append(steps)

    if done and steps < 500:
        successes += 1

    print(f"Episode {ep+1}: {steps} steps")

env.close()

print("\n=== MODEL PERFORMANCE ===")
print("Average steps:", np.mean(episode_steps))
print("Min steps:", np.min(episode_steps))
print("Max steps:", np.max(episode_steps))
print("Success rate:", successes / num_episodes)
