import time

import redis


class RemainingTimeHook:
    def __init__(self, max_iters: int):
        self.max_iters = max_iters
        self.start_time = time.time()
        self.iter_times = []
        self.r = redis.Redis(host='redis', port=6379, db=0)

    def update(self, iter: int):
        iter_time = time.time() - self.start_time
        self.iter_times.append(iter_time)

        avg_iter_time = sum(self.iter_times) / len(self.iter_times)

        total_iters = self.max_iters
        remaining_iters = total_iters - iter

        remaining_time = avg_iter_time * remaining_iters
        progress = ((iter + 1) / total_iters) * 100

        self.r.set("remaining_time", remaining_time / 60)
        self.r.set("progress", progress)

        self.start_time = time.time()
