from datetime import datetime

BATCH_SIZE = 32

MAX_ITERATIONS = 120000

DATA_LOADERS = {}

CUDA_VISIBLE_DEVICES = None

QUEUE_NUM_THREADS = 1
QUEUE_CAPACITY = 128

LOGNAME = datetime.now().strftime('%Y%m%d_%H:%M:%S')
