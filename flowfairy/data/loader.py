import random

class DataLoader:

    def __init__(self, name):
        self.name = name
        self._data = []

    def add_data(self, data):
        self._data.append(data)

    def _iter_data(self):
        while True:
            yield {d.name: next(d) for d in self._data}

    def __iter__(self):
        return self._iter_data()


