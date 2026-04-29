from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from load_data_mc import load_data
import evaluate_mc as evaluate

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

model = SVC(
    kernel="rbf"
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
