import config
from generator_model import Generator
from torch.utils.data import DataLoader
import torch.optim as optim
import utils
from dataset import ImgDataset
from os import path, mkdir

if not path.exists(config.TEST_DIR):
    mkdir(config.TEST_DIR)

G_test = Generator().to(config.DEVICE)
G_test_optim = optim.Adam(G_test.parameters(), lr=config.LR_GEN, betas=(config.BETA1, config.BETA2))
utils.load_checkpoint(config.CHECKPOINT_GEN, G_test, G_test_optim, config.LR_GEN)

val_or_test = input("Type 'val' for testing on random validation set sample or 'test' for custom B&W image test: ")

if val_or_test == "val":
    val_data = ImgDataset(img_dir=config.DATA_DIR, is_train=False)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=True)
    utils.save_results(G_test, val_loader, None)
    print("Check results sub-folder for saved images")

elif val_or_test == "test":
    TEST_IMG_PATH = input("Enter path of your own B&W testing image: ")
    IMG_NAME = input("Enter a name for your image: ")

    test_data = ImgDataset(img_dir=TEST_IMG_PATH, is_train=False, is_test=True)
    test_loader = DataLoader(test_data, batch_size=1)
    utils.test_an_image(G_test, test_loader, img_name=IMG_NAME, folder=config.SAVE_DIR)
    print("Check results sub-folder for saved images")

else: print("Incorrect value entered")