from training.optimizers.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, learningRate: float, momentumDecay: float = 0.9,
                 varianceDecay: float = 0.999, epsilon: float = 1e-8):
        self.learningRate = learningRate
        self.momentumDecay = momentumDecay
        self.varianceDecay = varianceDecay
        self.epsilon = epsilon
        self.updateCount = 1