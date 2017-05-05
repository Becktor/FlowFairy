import os

base_dir = os.path.dirname(os.path.realpath(__file__))

NET = 'net_classify'

FEATURES = [
    'feature.SineGen',
    'feature.NoisySineGen',
    'feature.Dropout',
    'feature.ConvertToClasses'
    #'feature.Mask'
]

SAMPLERATE = 11024//2
DURATION = 1
DROPOUT = 0.50
LEARNING_RATE = 0.001
CLASS_COUNT = 200
FREQUENCY_LIMIT = (120//2, 720//2)
EMBEDDING_SIZE = 1
EMBEDDING_INPUT_SIZE = 1
DISCRETE_CLASS = 256

SINE_COUNT = 2

BATCH_SIZE = 32
QUEUE_CAPACITY = 64 * BATCH_SIZE

CUDA_VISIBLE_DEVICES = 3

LOG_INTERVAL = 100//BATCH_SIZE

LOG_DIR = os.path.join(base_dir, "logs")

DATA = {
    'train': (),
    'validation': (),
    'test': ()
}
