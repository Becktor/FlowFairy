from flowfairy.feature import Feature
from flowfairy.conf import settings
import numpy as np


samplerate = settings.SAMPLERATE
duration = settings.DURATION
chunk = settings.CHUNK


class SineGen(Feature):
    arr = np.arange(samplerate, dtype=np.float32) * 2 * np.pi / samplerate

    def feature(self, frequencies, **kwargs):
        frq = frequencies

        y = np.sin(self.arr * frq[1]).astype('float32')
        return {'y': y}

    class Meta:
        ignored_fields = ('frequencies','blends')


class NoisySineGen(Feature):

    def feature(self, y, **kwargs):
        noise = np.random.uniform(-0.5, 0.5, samplerate).astype('float32')
        return {'x': noise + y}

    def fields(self):
        return ('x',)

class ConvertToClasses(Feature):

    def feature(self, x, y, **kwargs):
        return {'x':x, 'y':y}

class Chunk(Feature):

    def feature(self, x, **kwargs):
        k = np.ones(samplerate * duration, dtype=np.float32)
        j = np.random.randint(chunk, samplerate*duration)
        k[j:j+chunk] = 0
        return {'x':x*k, 'chunk':np.array(j)}

    def fields(self):
        return ('chunk',)

class Dropout(Feature):

    def feature(self, **kwargs):
        return {'keep_prob': np.array(0.50, dtype=np.float32)}

    def fields(self):
        return ('keep_prob',)
