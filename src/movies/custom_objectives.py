import numpy as np

class AsymmetricEdgePenaltyObjective:
    """
    Implements the asymmetric edge-penalty loss:
    L(e, r) = 0.5 * e² · exp(alpha_hi * max(0, r - 4.0) + alpha_lo * max(0, 1.5 - r))
    where e = pred - true.
    
    This penalizes errors more heavily at the extremes (Must Watch / Hard Pass).
    """
    def __init__(self, alpha_hi=0.1, alpha_lo=0.1):
        self.alpha_hi = alpha_hi
        self.alpha_lo = alpha_lo
        
    def __call__(self, y_true, y_pred, sample_weight=None):
        # Calculate weights based on the true rating
        # High penalty for errors on items you love (r >= 4.0)
        # Low penalty for errors on items you hate (r <= 1.5)
        W_y = np.exp(
            self.alpha_hi * np.maximum(0, y_true - 4.0) + 
            self.alpha_lo * np.maximum(0, 1.5 - y_true)
        )
        
        if sample_weight is not None:
            W_y = W_y * sample_weight
            
        # Gradient = W_y * (y_pred - y_true)
        grad = W_y * (y_pred - y_true)
        
        # Hessian = W_y
        hess = W_y * np.ones_like(y_true)
        
        return grad, hess

class CustomEdgePenaltyObjective:
    """
    Legacy class for compatibility.
    """
    def __init__(self, alpha=1.0, mu=3.0):
        self.alpha = alpha
        self.mu = mu
        
    def __call__(self, y_true, y_pred):
        W_y = np.exp(self.alpha * np.abs(y_true - self.mu))
        grad = W_y * (y_pred - y_true)
        hess = W_y * np.ones_like(y_true)
        return grad, hess
