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
# LOAD ALL FILES
# =========================

files = glob.glob(PATTERN)

print("\nFound files:")
for f in files:
    print(" -", f)

dfs = []

for f in files:
    try:
        df = pd.read_csv(f)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
        print(f"Loaded {f} with {len(df)} rows")
    except Exception as e:
        print(f"Failed to load {f}: {e}")

if not dfs:
    raise RuntimeError("No data loaded.")

df = pd.concat(dfs, ignore_index=True)

print("\nTotal rows:", len(df))

# =========================
# PARSE STATE
# =========================

def parse_state(s):
    try:
        if isinstance(s, str):
            return np.array(json.loads(s), dtype=float)
        return None
    except:
        return None

df["state_parsed"] = df["state"].apply(parse_state)

# drop bad rows
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

# safety check (prevents your crash)
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

print("\n=== RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

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

print("\nAverage episode length:")
print(df.groupby("episode")["step"].max().mean())

print("\nSuccess rate:")
if "success" in df.columns:
    print(df["success"].mean())
