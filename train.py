import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load data
df = pd.read_csv("diabetes.csv")

# 2. Separate features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# 3. 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. RandomForest model
clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)

# 5. Train
clf.fit(X_train, y_train)

# 6. Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.4f}")

# 7. Save model
joblib.dump(clf, "model.pkl")
print("Saved model to model.pkl")

