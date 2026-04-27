import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def run(model, scaler, X_train, X_test, y_train, y_test, df):

    print("\n=== TEST RESULTS ===")
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix (TEST):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\n=== TRAIN RESULTS ===")
    y_pred_train = model.predict(X_train)

    print("Accuracy:", accuracy_score(y_train, y_pred_train))

    print("\nConfusion Matrix (TRAIN):")
    print(confusion_matrix(y_train, y_pred_train))

    print("\nAverage human episode length:")
    print(df.groupby("episode")["step"].max().mean())

    print("\nSuccess rate:")
    print(df["success"].mean())

    # optional rollout
    try:
        import gymnasium as gym

        print("\n=== MODEL ROLLOUT (5 EPISODES) ===")

        env = gym.make("Acrobot-v1")

        def obs_to_state(obs):
            return np.array(obs, dtype=float)

        steps_list = []
        successes = 0

        for ep in range(5):
            obs, _ = env.reset()
            done = False
            steps = 0

            while not done and steps < 500:

                state = obs_to_state(obs).reshape(1, -1)
                state = scaler.transform(state)

                action = model.predict(state)[0]

                obs, reward, terminated, truncated, _ = env.step(int(action))
                done = terminated or truncated
                steps += 1

            steps_list.append(steps)

            if terminated:
                successes += 1

            print(f"Episode {ep+1}: {steps} steps")

        env.close()

        print("\nAvg steps:", np.mean(steps_list))

    except Exception as e:
        print("Rollout failed:", e)
