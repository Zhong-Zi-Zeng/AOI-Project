from threading import Thread

import redis

class TrainingManager:
    def __init__(self):
        self.thread = None
        self.complete = False
        self.r = redis.Redis(host='redis', port=6379, db=0)

    def start_training(self, train_func):
        self.complete = False
        self.thread = Thread(target=self._train_wrapper, args=(train_func,))
        self.thread.start()

    def _train_wrapper(self, train_func):
        train_func()
        self.complete = True
        self._clear_redis_key()

    def _clear_redis_key(self):
        self.r.delete("remaining_time")

    def get_remaining_time(self):
        return self.r.get("remaining_time")

    def is_training(self):
        return self.thread is not None and self.thread.is_alive()

    def is_complete(self):
        return self.complete
