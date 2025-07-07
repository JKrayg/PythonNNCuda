import cupy as cp

from training.loss.loss import Loss


class CatCrossEntropy(Loss):
    def __init__(self):
        print("cat cross entropy")

    def execute(activation: cp.ndarray, labels: cp.ndarray):
        error = labels * cp.log(activation)

        return -(cp.sum(error) / activation.shape[0])
    
    def gradient(activation: cp.ndarray, labels: cp.ndarray):
        return activation - labels