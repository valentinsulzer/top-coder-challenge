import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso
import xgboost as xgb
import numpy as np
# from human_loop_model import HumanLoopModel


class EnsembleModel:
    """
    An ensemble model that averages predictions from multiple models.
    It's treated as a custom model in this framework.
    """

    def __init__(self, models=None):
        self.models_to_train = models if models else self._get_default_models()
        self.trained_models = []

    def _get_default_models(self):
        return [get_lgbm_model(), get_rf_model(), get_xgb_model()]

    def fit(self, X, y):
        for model in self.models_to_train:
            print(f"  - Training {model.__class__.__name__}...")
            try:
                model.fit(X, y, verbose=False)
            except TypeError:
                model.fit(X, y)
            self.trained_models.append(model)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.trained_models])
        return np.mean(predictions, axis=0)

    # This is needed to be compatible with the scikit-learn API used in the runner
    def get_params(self, deep=True):
        return {}


def get_lgbm_model():
    return lgb.LGBMRegressor(
        objective="regression_l1",
        metric="mae",
        n_estimators=2000,
        learning_rate=0.01,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        lambda_l1=0.1,
        lambda_l2=0.1,
        num_leaves=31,
        verbose=-1,
        n_jobs=-1,
        seed=42,
        boosting_type="gbdt",
    )


def get_rf_model():
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def get_xgb_model():
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        min_child_weight=5,
        n_jobs=-1,
        seed=42,
    )


def get_et_model():
    return ExtraTreesRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )


def get_ensemble_model():
    return EnsembleModel()


def get_ridge_model():
    return Ridge(alpha=1.0, random_state=42)


def get_lasso_model():
    return Lasso(alpha=0.1, random_state=42)


MODEL_CONFIGS = {
    "lightgbm": {"model": get_lgbm_model, "name": "LightGBM"},
    "random_forest": {"model": get_rf_model, "name": "Random Forest"},
    "xgboost": {"model": get_xgb_model, "name": "XGBoost"},
    "extra_trees": {"model": get_et_model, "name": "Extra Trees"},
    "ridge": {"model": get_ridge_model, "name": "Ridge Regression"},
    "lasso": {"model": get_lasso_model, "name": "Lasso Regression"},
    "ensemble": {"model": get_ensemble_model, "name": "Ensemble"},
    # "human_loop": {"model": get_human_loop_model, "name": "Human-in-the-Loop"},
}
