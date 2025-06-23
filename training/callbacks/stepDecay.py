from training.callbacks.callback import Callback


class StepDecay(Callback):
    def __init__(self):
        print("step decay")