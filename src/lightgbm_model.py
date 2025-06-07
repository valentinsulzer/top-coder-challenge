import lightgbm as lgb
import pandas as pd


class LightGBMReimbursementModel:
    def __init__(self):
        self.model = None
        self.feature_names = None

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        self.model = lgb.LGBMRegressor(random_state=42)
        self.model.fit(X, y)

    def predict(self, X):
        # Ensure columns match training
        for col in self.feature_names:
            if col not in X:
                X[col] = 0
        X = X[self.feature_names]
        return self.model.predict(X)
