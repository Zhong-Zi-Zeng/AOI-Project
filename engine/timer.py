import contextlib
import torch
import time

class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, name: str, t=0.0):
        self.name = name
        self.t = t
        self.dt = 0
        self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = (self.time() - self.start) * 1000  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()


TIMER = (Profile(name='Total process time(ms):'),
         Profile(name='Preprocess Time(ms):'),
         Profile(name='Inference Time(ms):'),
         Profile(name='NMS Time(ms):'))
