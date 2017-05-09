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
frqs2 = list(frqs)
random.shuffle(frqs2)
frqs3 = list(frqs)
random.shuffle(frqs3)
frqs4 = list(frqs)
random.shuffle(frqs4)
frqs5 = list(frqs)
random.shuffle(frqs5)

def frequencies():
    return it.chain(zip(frqs, frqs2), zip(frqs, frqs3), zip(frqs, frqs4), zip(frqs, frqs5))


data.register(frequencies, ('train', 0.75), ('validation', 0.25))
