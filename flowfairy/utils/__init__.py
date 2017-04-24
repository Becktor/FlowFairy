import importlib
from itertools import islice

def import_from_module(im):
    try:
        components = im.split('.')
        module = importlib.import_module('.'.join(components[:-1]))
        return getattr(module, components[-1])
    except ImportError:
        raise ImportError(f'Could not import {im}')

def dyncall(istr, *args, **options):
    return import_from_module(istr)(*args, **options)


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))
