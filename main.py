import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# Simulated data
X = np.random.randn(100, 3)
y = np.random.randint(0, 2, 100)

model = LogisticRegression().fit(X, y)
print("Model trained. Coefficients:", model.coef_)

df = pd.DataFrame(X, columns=["feature1", "feature2", "feature3"])
df["target"] = y
print(df.head())
