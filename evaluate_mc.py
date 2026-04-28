import numpy as np

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import gymnasium as gym


def run(model, scaler,
        X_train, X_test,
        y_train, y_test,
        df):

    # =========================
    # TEST RESULTS
    # =========================

    y_pred = model.predict(X_test)

    print("\n=== TEST RESULTS ===")

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix (TEST)")
    print("Rows = TRUE")
    print("Cols = PRED")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # =========================
    # TRAIN RESULTS
    # =========================

    y_pred_train = model.predict(X_train)

    print("\n=== TRAIN RESULTS ===")

    print("Accuracy:",
          accuracy_score(y_train, y_pred_train))

    print("\nConfusion Matrix (TRAIN)")
    print(confusion_matrix(y_train, y_pred_train))

    # =========================
    # BASELINES
    # =========================

    random_preds = np.random.choice(
        np.unique(y_test),
        size=len(y_test)
    )

    majority = np.bincount(y_train).argmax()

    majority_preds = np.full_like(
        y_test,
        majority
    )

    print("\n=== BASELINES ===")

    print("Random:",
          accuracy_score(y_test, random_preds))

    print("Majority:",
          accuracy_score(y_test, majority_preds))

    # =========================
    # HUMAN METRICS
    # =========================

    print("\n=== HUMAN PERFORMANCE ===")

    print("Average episode length:")
    print(df.groupby("episode")["step"].max().mean())

    print("\nSuccess rate:")
    print(df["done"].mean())

    # =========================
    # MODEL ROLLOUT
    # =========================

    print("\n=== MODEL ROLLOUT (5 EPISODES) ===")

    env = gym.make("MountainCar-v0")

    episode_steps = []
    successes = 0

    for ep in range(5):

        obs, _ = env.reset()

        done = False
        steps = 0

        while not done and steps < 500:

            state = np.array(obs).reshape(1, -1)

            state = scaler.transform(state)

            action = model.predict(state)[0]

            obs, reward, terminated, truncated, _ = env.step(int(action))

            done = terminated or truncated

            steps += 1

            # success condition
            if obs[0] >= 0.5:
                successes += 1
                break

        episode_steps.append(steps)

        print(f"Episode {ep+1}: {steps} steps")

    env.close()

    print("\n=== MODEL PERFORMANCE ===")

    print("Average steps:",
          np.mean(episode_steps))

    print("Min steps:",
          np.min(episode_steps))

    print("Max steps:",
          np.max(episode_steps))

    print("Success rate:",
          successes / 5)
