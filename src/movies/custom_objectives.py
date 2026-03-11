import numpy as np

class CustomEdgePenaltyObjective:
    """
    Callable class that acts as a custom XGBoost objective.
    Because it is a class, it can be parameterized (alpha) 
    AND it remains fully picklable for joblib and StackingRegressor.
    """
    def __init__(self, alpha=1.0, mu=3.0):
        self.alpha = alpha
        self.mu = mu
        
    def __call__(self, y_true, y_pred):
        # Calculate exponential weight based on distance from the mean
        W_y = np.exp(self.alpha * np.abs(y_true - self.mu))
        
        # Apply to gradients and hessians
        grad = W_y * (y_pred - y_true)
        hess = W_y * np.ones_like(y_true)
        
        return grad, hess