from training.callbacks.callback import Callback


class EarlyStopping(Callback):
    def __init__(self):
        print("early stopping")