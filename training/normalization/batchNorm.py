from training.normalization.normalizer import Normalizer


class BatchNormalization(Normalizer):
    def __init__(self):
        self.shift = None
        self.scale = None
        self.means = None
        self.variances = None
        self.runMeans = None
        self.runVar = None