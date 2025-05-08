# Path setting
DATASET_PATH = "/Users/dg/DDP/chord_generation-main_v3_metrics/dataset"
CORPUS_PATH = "/Users/dg/DDP/chord_generation-main_v3_metrics/corpus.bin"
WEIGHTS_PATH = '/Users/dg/DDP/chord_generation-main_v3_metrics/weights.keras'
INPUTS_PATH = "/Users/dg/DDP/chord_generation-main_v3_metrics/inputs"
OUTPUTS_PATH = "/Users/dg/DDP/chord_generation-main_v3_metrics/outputs"

# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl', '.midi', '.mid']

# 'train_model.py'
VAL_RATIO = 0.1
RNN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 512
EPOCHS = 40

# 'harmonizor.py'
TEMPERATURE = 0
RHYTHM_DENSITY = 0
LEAD_SHEET = True