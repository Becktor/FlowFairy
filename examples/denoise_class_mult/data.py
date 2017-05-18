import numpy as np
import random
import itertools as it

from flowfairy import data
from flowfairy.conf import settings


samplerate = settings.SAMPLERATE
frequency_count = settings.CLASS_COUNT
frq_min, frq_max = settings.FREQUENCY_LIMIT
step = (frq_max - frq_min) / frequency_count

frqs = list(enumerate(np.arange(frq_min, frq_max, step)))
blnds = list(frqs)
random.shuffle(blnds)

val_cut = int(frequency_count * 0.75)

def frequencies():
    while True:
        random.shuffle(frqs)
        yield from frqs

def blends_train():
    blends = list(blnds[:val_cut])
    while True:
        random.shuffle(blends)
        yield from blends

def blends_val():
    blends = list(blnds[val_cut:])
    while True:
        random.shuffle(blends)
        yield from blends

data.register(frequencies, 'train')
data.register(frequencies, 'validation')

data.register(blends_train, 'train', name='blends')
data.register(blends_val, 'validation', name='blends')
