import random

from .loader import DataLoader

class Data:

    def __init__(self, name, data):
        self.name = name
        self.data = data
        self.iterator = self.iter_data()

    def iter_data(self):
        while True:
            yield from self.data()

    def __next__(self):
        return next(self.iterator)


class _DataProvider:

    def __init__(self):
        self.loaders = {}

    def register(self, data, data_set):
        loader = self.loaders.get(data_set, None)
        if not loader:
            loader = DataLoader(name=data_set)
            self.loaders[data_set] = loader

        loader.add_data(data)

    def __iter__(self):
        return iter(self.loaders.values())


provider = _DataProvider()

def register(loader_func, data_set, *, name=None):
    name = name or loader_func.__name__

    data = Data(name, loader_func)
    provider.register(data, data_set)


try:
    import data
except ImportError:
    raise ImportError('Need some sort of data. Call flowfairy.data.register in data.py.')
