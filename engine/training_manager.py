from threading import Thread


class TrainingManager:
    def __init__(self):
        self.thread = None
        self.complete = False

    def start_training(self, train_func):
        self.complete = False
        self.thread = Thread(target=self._train_wrapper, args=(train_func,))
        self.thread.start()

    def _train_wrapper(self, train_func):
        train_func()
        self.complete = True

    def is_training(self):
        return self.thread is not None and self.thread.is_alive()

    def is_complete(self):
        return self.complete
