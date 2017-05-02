import random

from .loader import DataLoader

class _DataProvider:

    def __init__(self):
        self.loaders = []

    def register(self, loader_func, *sets):
        if not sets:
            raise ValueError('Need to specify name of dataset')

        all_data = list(loader_func())
        random.shuffle(all_data)

        if len(sets) == 1:
            self.loaders.append(DataLoader(name=sets[0], data=all_data))
            return

        total_frac = sum(map(lambda nf: nf[1], sets))
        size = len(all_data)
        idx = 0

        for name, frac in sets:
            frac = frac / total_frac
            end = int(idx + size * frac)

            data = all_data[idx:end]

            self.loaders.append(DataLoader(name=name, data=data))

            idx = end

    def __iter__(self):
        return iter(self.loaders)

provider = _DataProvider()

def register(loader, *sets):
    provider.register(loader, *sets)

try:
    import data
except:
    raise ImportError('Need some sort of data. Call flowfairy.data.register in data.py.')
