from training.callbacks.callback import Callback


class LRScheduler(Callback):
    def __init__(self):
        print("lrscheduler")