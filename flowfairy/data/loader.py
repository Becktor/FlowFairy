import random

class DataLoader:

    def __init__(self, name, data):
        self.name = name
        self.data = data

    def _iter_data(self):
        while True:
            random.shuffle(self.data)
            for d in self.data:
                yield {'initial': d}

    def __iter__(self):
        return self._iter_data()

