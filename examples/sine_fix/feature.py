from flowfairy.feature import Feature
from flowfairy.conf import settings

import numpy as np


samplerate = settings.SAMPLERATE
duration = settings.DURATION


class FrequencyGen(Feature):

    frqs = np.arange(340, 720, 40)

    def feature(self):
        return {'frequency': np.random.choice(self.frqs)}


class SineGen(Feature):

    def feature(self, frequency, **kwargs):
        arr = np.arange(samplerate * duration)
        return {'y': (np.sin(2 * np.pi * arr).astype('float32'))}

    def fields(self):
        return ('y',)


class NoisySineGen(Feature):

    def feature(self, y, **kwargs):
        noise = np.random.uniform(-0.5, 0.5, samplerate)
        return {'x': ( noise+y ).astype('float32')}

    def fields(self):
        return ('x',)


class Mask(Feature):

    def feature(self, **kwargs):
        return {'m': np.ones(samplerate * duration).astype('float32')}

    def fields(self):
        return ('m',)



class Dropout(Feature):

    def feature(self, **kwargs):
        return {'keep_prob': np.array(0.50, dtype=np.float32)}

    def fields(self):
        return ('keep_prob',)
