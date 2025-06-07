import numpy as np
from scipy.optimize import minimize
from human_model import output_model


class HumanLoopModel:
    """
    A model that uses an optimization algorithm to fit the parameters of the human_model.
    """

    def __init__(self, n_params=28):
        self.n_params = n_params
        self.params = np.random.rand(n_params)

    def _objective(self, params, X, y):
        # For this model, we'll only use the three core features
        mileage = X["miles_traveled"]
        receipts = X["total_receipts_amount"]
        duration = X["trip_duration_days"]

        # Calculate predictions using the human_model's output_model
        predictions = output_model(params, mileage, receipts, duration)

        # The loss is the mean squared error
        loss = np.mean((y - predictions) ** 2)
        return loss

    def fit(self, X, y):
        print("Optimizing human model parameters...")

        # We need to denormalize the target variable `y` since it's log-transformed
        # and represents reimbursement per day.
        duration_safe = X["trip_duration_days"].replace(0, 1)
        y_denormalized = np.expm1(y) * duration_safe

        # Run the optimization
        result = minimize(
            self._objective,
            self.params,
            args=(X, y_denormalized),
            method="Nelder-Mead",
            options={"maxiter": 500, "disp": True},
        )

        # Store the best-found parameters
        self.params = result.x
        print("Optimization complete.")

    def predict(self, X):
        mileage = X["miles_traveled"]
        receipts = X["total_receipts_amount"]
        duration = X["trip_duration_days"]

        # Generate predictions with the optimized parameters
        predictions = output_model(self.params, mileage, receipts, duration)

        # We need to re-normalize the predictions to match the expected log-transformed output
        duration_safe = duration.replace(0, 1)
        reimbursement_per_day = predictions / duration_safe
        return np.log1p(reimbursement_per_day)

    def get_params(self, deep=True):
        # Required for compatibility with scikit-learn's API
        return {"n_params": self.n_params}
