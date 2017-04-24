import tensorflow as tf
import threading
from flowfairy.conf import settings
from flowfairy.utils import import_from_module, take

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

class Feature:

    @property
    def dtypes(self):
        raise NotImplementedError("Features need a dtype property")

    def feature(*args, **options):
        raise NotImplementedError

    def fields(self):
        return ()

    def shapes(self):
        pass


class FeatureManager:

    def __init__(self, dataloader):
        self.dataloader = dataloader

        features = settings.FEATURES
        self.features = [import_from_module(s)() for s in features]

        self._data_gen = self.get_features()
        self._latest = next(self._data_gen)
        self._placeholders = None
        print(self.fields)
        print(self.dtypes)

    def _get_shapes(self):
        for l, field in zip(self._latest, self.fields):
            try:
                feature = first_true(self.features, default=None, pred=lambda ft: field in ft.fields())
                meta = getattr(feature, 'Meta')
                shapes = getattr(meta, 'shapes')
                yield shapes[field]
            except:
                yield l.shape

    @property
    def shapes(self):
        return list(self._get_shapes())

    @property
    def dtypes(self):
        return [tf.as_dtype(f.dtype) for f in self._latest]

    @property
    def fields(self):
        return list(it.chain(*[ft.fields() for ft in self.features]))

    def get_features(self):
        for data in self.dataloader:

            for feature in self.features:
                data.update(feature.feature(**data))

            # Filter the dict so we only get what we want
            yield [data[key] for key in self.fields]

    def _init_placeholders(self):
        self._placeholders = [
            tf.placeholder(dtype, shape, name=name) for dtype, shape, name in zip(self.dtypes, self.shapes, self.fields)
        ]

    @property
    def placeholders(self):
        if not self._placeholders:
            self._init_placeholders()
        return self._placeholders

    def __iter__(self):
        return self

    def __next__(self):
        latest = self._latest
        self._latest = next(self._data_gen)
        return latest

    def batch(self, bs):
        return [take(bs, feature) for feature in next(self)]


