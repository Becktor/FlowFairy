import tensorflow as tf
import threading
import numpy as np

from flowfairy.conf import settings
from flowfairy.utils import import_from_module, take

from .base import Feature

import itertools as it

def listchain(*args):
    return list(it.chain(*args))


def first_true(iterable, default=False, pred=None):
    """Returns the first true value in the iterable.

    If no true value is found, returns *default*

    If *pred* is not None, returns the first item
    for which pred(item) is true.

    """
    # first_true([a,b,c], x) --> a or b or c or x
    # first_true([a,b], x, f) --> a if f(a) else b if f(b) else x
    return next(filter(pred, iterable), default)


class FeatureManager:

    def __init__(self, dataloader):
        self.dataloader = dataloader

        features = settings.FEATURES

        self.features = [import_from_module(s)() for s in features]

        self._data_gen = self.get_features()
        self._latest = next(self._data_gen)
        self._ignored_fields = set(self._get_ignored())
        self._ensure_types()
        print(self.fields)
        print(self.dtypes)

    def _get_ignored(self):
        for ft in self.features:
            for ignored in ft._meta.ignored_fields:
                yield ignored

    def _get_shapes(self):
        try:
            for l, field in zip(self.filtered(), self.fields):
                yield l.shape
        except Exception as e:
            if field:
                print(e)
                raise ValueError(f"Got a problem with {field}")

    @property
    def shapes(self):
        return list(self._get_shapes())

    type_error_string = 'The feature {field} needs to have a {attr}. Consider wrapping it with np.array().'
    def _ensure_types(self):
        for f, field in zip(self.filtered(), self.fields):
            if not hasattr(f, 'dtype'):
                raise ValueError(self.type_error_string.format(field=field, attr='dtype'))
            if not hasattr(f, 'shape'):
                raise ValueError(self.type_error_string.format(field=field, attr='shape'))

    @property
    def dtypes(self):
        return [tf.as_dtype(f.dtype) for f in self.filtered()]

    @property
    def fields(self):
        return set(self._latest.keys()) - self._ignored_fields

    def filtered(self):
        # Filter the dict so we only get what we want
        return [self._latest[key] for key in self.fields]

    def _run_features(self, initial_data):
        data = {}
        for feature in self.features:
            data.update(feature.feature(**data, **initial_data))
            initial_data = {k:v for k,v in initial_data.items() if k not in data}

        return data

    def get_features(self):
        for initial_data in self.dataloader:
            try:
                data = self._run_features(initial_data)
            except FeatureError:
                continue

            yield data

    def __iter__(self):
        return self

    def __next__(self):
        latest = self._latest
        self._latest = next(self._data_gen)
        return self.filtered()

class FeatureError(Exception):
    pass
