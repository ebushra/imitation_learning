from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from data_loader import load_data

df, X, y = load_data()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
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

import evaluate
evaluate.run(model, scaler, X_train, X_test, y_train, y_test, df)
