from flowfairy.conf import settings
from flowfairy.utils import import_from_module

import random

class DataLoader:

    def __init__(self):
        loaders = settings.DATA_LOADERS
        self._loaders = {k: import_from_module(v) for k, v in loaders.items()}

    def _get_data_from_loader(self, loader):
        data = list(loader(settings.DATA_DIRECTORY))

        while True:
            random.shuffle(data)
            for d in data:
                yield d

    def __iter__(self):
        self._data = {k: self._get_data_from_loader(v) for k, v in self._loaders.items()}
        return self

    def __next__(self):
        return {k: next(v) for k, v in self._data.items()}


