import joblib
import numpy as np

clf = joblib.load("model.pkl")

sample = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])

pred = clf.predict(sample)[0]
proba = clf.predict_proba(sample)[0][1]

print(f"Predicted class (1=diabetic, 0=non-diabetic): {pred}")
print(f"Probability of diabetes: {proba:.4f}")

