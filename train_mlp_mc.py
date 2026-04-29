from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from load_data_mc import load_data
import evaluate_mc

df, X, y = load_data()

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

model = MLPClassifier(
    hidden_layer_sizes=(64, 64),
    max_iter=200,
    random_state=42
)

model.fit(X_train, y_train)

evaluate.run(
    model,
    scaler,
    X_train,
    X_test,
    y_train,
    y_test,
    df
)
