import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        
        x, W1, b1, W2, b2 = np.asarray(x,float), np.asarray(W1,float), np.asarray(b1,float), np.asarray(W2,float), np.asarray(b2,float) 

        z1 = x @ W1.T + b1
        a1 = np.maximum(z1,0)
        z2 = a1 @ W2.T + b2
        
        loss = np.mean((z2 - y_true)**2)
        
        dz2 = 2 / len(y_true) * (z2 - y_true)
        db2 = dz2
        dW2 = np.outer(dz2,a1)
        
        da1 = dz2 @ W2
        dz1 = da1 * (z1 > 0)
        db1 = dz1
        dW1 = np.outer(dz1,x)

        return {
            "loss": round(float(loss), 4),
            "dW1": np.round(dW1, 4).tolist(),
            "db1": np.round(db1, 4).tolist(),
            "dW2": np.round(dW2, 4).tolist(),
            "db2": np.round(db2, 4).tolist(),
        }