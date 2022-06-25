from torch import device, cuda

DEVICE = device("cuda:0" if cuda.is_available() else "cpu")
LR_GEN = 2e-4
LR_DISC = 2e-4
BETA1 = 0.5
BETA2 = 0.999
NUM_WORKERS = 2
BATCH_SIZE = 20
NUM_EPOCHS = 2
L1_LAMBDA = 100
DATA_DIR = "./data/train/"
SAVE_DIR = "./results/"
MODEL_DIR = "./model_params/"
LOG_DIR = "./log_dir/"
CHECKPOINT_DISC = "./model_params/disc.pth.tar"
CHECKPOINT_GEN = "./model_params/gen.pth.tar"
TEST_DIR = "./test"
SAVE_MODEL = False
LOAD_MODEL = True
