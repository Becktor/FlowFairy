import numpy as np
import random

from flowfairy import data
from flowfairy.conf import settings


samplerate = settings.SAMPLERATE
frequency_count = settings.CLASS_COUNT
frq_min, frq_max = settings.FREQUENCY_LIMIT
step = (frq_max - frq_min) / frequency_count

frqs = list(np.arange(frq_min, frq_max, step))

def frequencies():
    return frqs


data.register(frequencies, ('train', 0.75), ('validation', 0.25))
