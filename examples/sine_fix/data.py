import numpy as np
import random

from flowfairy import data
from flowfairy.conf import settings


samplerate = settings.SAMPLERATE
frequency_count = settings.CLASS_COUNT
frq_min, frq_max = settings.FREQUENCY_LIMIT
step = (frq_max - frq_min) / frequency_count

frqs = list(enumerate(np.arange(frq_min, frq_max, step)))
frqs2 = list(frqs)
random.shuffle(frqs2)

def frequencies():
    return zip(frqs, frqs2)


data.register(frequencies, ('train', 0.75), ('validation', 0.25))
