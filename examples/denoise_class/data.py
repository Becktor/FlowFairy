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

val_cut = int(frequency_count * 0.75)

def frequencies():
    while True:
        random.shuffle(frqs)
        yield from frqs

data.register(frequencies, 'train')
data.register(frequencies, 'validation')
