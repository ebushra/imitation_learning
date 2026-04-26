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
# PARSE STATE
# =========================

def parse_state(s):
    try:
        if not isinstance(s, str):
            return None
        return np.array(json.loads(s), dtype=float)
    except:
        return None

df["state_parsed"] = df["state"].apply(parse_state)

# =========================
# CLEAN DATA
# =========================

df = df.dropna(subset=["state_parsed"])

# REMOVE TRAINING DATA
if "training" in df.columns:
    df = df[df["training"] == False]

print("After removing training rows:", len(df))

# REMOVE UNFINISHED EPISODES
if "success" in df.columns:
    finished_eps = df.groupby("episode")["success"].max()
    finished_eps = finished_eps[finished_eps == True].index
    df = df[df["episode"].isin(finished_eps)]

print("After removing unfinished episodes:", len(df))

# =========================
# BUILD DATASET
# =========================

X = np.vstack(df["state_parsed"].values)
y = df["action"].astype(int).values

print("\nDataset shape:", X.shape)
print("Action distribution:", np.bincount(y))

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

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

print("\n=== TEST RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred_test))

print("\nConfusion Matrix (Test):")
print("Rows = TRUE, Cols = PRED")
print("      0   1   2")
print(confusion_matrix(y_test, y_pred_test))

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test))

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
# HUMAN PERFORMANCE
# =========================

print("\n=== HUMAN PERFORMANCE ===")

avg_steps = df.groupby("episode")["step"].max().mean()
print("Average human steps:", avg_steps)

if "success" in df.columns:
    print("Success rate:", df["success"].mean())

# =========================
# MODEL STEP ESTIMATE (NEW)
# =========================

print("\n=== MODEL STEP ESTIMATE ===")

# approximate: how long model would take given its mistakes
test_steps = df.iloc[y_test.index]["step"].values

# if correct → same step
# if wrong → penalize (simulate inefficiency)
penalty = 5  # tweak this if you want stricter penalty

model_steps = []
for i in range(len(y_test)):
    if y_pred_test[i] == y_test[i]:
        model_steps.append(test_steps[i])
    else:
        model_steps.append(test_steps[i] + penalty)

print("Estimated model steps:", np.mean(model_steps))
