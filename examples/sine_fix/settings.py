import os

base_dir = os.path.dirname(os.path.realpath(__file__))

NET = 'gglu5'

FEATURES = [
    'feature.SineGen',
    'feature.NoisySineGen',
    #'feature.Dropout',
    'feature.ConvertToClasses'
    #'feature.Mask'
]

SAMPLERATE = 2**14
DURATION = 1
DROPOUT = 0.50
LEARNING_RATE = 0.001
CLASS_COUNT = 200
FREQUENCY_LIMIT = (120, 720)
EMBEDDING_SIZE = 2
DISCRETE_CLASS = 256
MAX_AMP = 5


BATCH_SIZE = 32
QUEUE_CAPACITY = 4 * BATCH_SIZE

CUDA_VISIBLE_DEVICES = 0

LOG_INTERVAL = 100//BATCH_SIZE

LOG_DIR = os.path.join(base_dir, "logs")

DATA = {
    'train': (),
    'validation': (),
    'test': ()
}
