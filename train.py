import torch
import config
import utils
from logger import Logger
from dataset import ImgDataset
from generator_model import Generator
from discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

if not os.path.exists(config.SAVE_DIR):
    os.mkdir(config.SAVE_DIR)
if not os.path.exists(config.MODEL_DIR):
    os.mkdir(config.MODEL_DIR)
if not os.path.exists(config.LOG_DIR):
    os.mkdir(config.LOG_DIR)

def train_fn(G, D, G_optim, D_optim, G_scaler, D_scaler, loader, L1, bce, G_logger, D_logger, step):
    D_losses = []
    G_losses = []
    loop = tqdm(loader, leave=True)

    for data in loop:
        Ls, ABs = data['L'], data['ab']
        Ls = Ls.to(config.DEVICE)
        ABs = ABs.to(config.DEVICE)

        # Discriminator training
        with torch.cuda.amp.autocast():
            AB_gen = G(Ls)
            D_real = D(Ls, ABs)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = D(Ls, AB_gen.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) * 0.5

        D.zero_grad()
        D_scaler.scale(D_loss).backward()
        D_scaler.step(D_optim)
        D_scaler.update()

        # Generator Training
        with torch.cuda.amp.autocast():
            D_fake = D(Ls, AB_gen)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1_loss = L1(AB_gen, ABs) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1_loss

        G.zero_grad()
        G_scaler.scale(G_loss).backward()
        G_scaler.step(G_optim)
        G_scaler.update()

        D_losses.append(D_loss.data)
        G_losses.append(G_loss.data)

        # TensorBoard logging
        D_logger.scalar_summary('D_losses', D_loss.data, step + 1)
        G_logger.scalar_summary('G_losses', G_loss.data, step + 1)
        step += 1

    D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
    G_avg_loss = torch.mean(torch.FloatTensor(G_losses))
    print('D_avg_loss: %.4f, G_avg_loss: %.4f' % (D_avg_loss.data, G_avg_loss.data))


def main():
    D = Discriminator().to(config.DEVICE)
    G = Generator().to(config.DEVICE)
    D_optim = torch.optim.Adam(D.parameters(), lr=config.LR_DISC, betas=(config.BETA1, config.BETA2))
    G_optim = torch.optim.Adam(G.parameters(), lr=config.LR_GEN, betas=(config.BETA1, config.BETA2))
    BCE = torch.nn.BCEWithLogitsLoss()
    L1_LOSS = torch.nn.L1Loss()

    D_log_dir = config.LOG_DIR + 'D_logs'
    G_log_dir = config.LOG_DIR + 'G_logs'
    if not os.path.exists(D_log_dir):
        os.mkdir(D_log_dir)
    if not os.path.exists(G_log_dir):
        os.mkdir(G_log_dir)
    D_logger = Logger(D_log_dir)
    G_logger = Logger(G_log_dir)

    if config.LOAD_MODEL:
        utils.load_checkpoint(config.CHECKPOINT_GEN, G, G_optim, config.LR_GEN)
        utils.load_checkpoint(config.CHECKPOINT_DISC, D, D_optim, config.LR_DISC)

    train_data = ImgDataset(img_dir=config.DATA_DIR)
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_data = ImgDataset(img_dir=config.DATA_DIR, is_train=False)
    val_loader = DataLoader(val_data, batch_size=5, shuffle=False)

    G_scaler = torch.cuda.amp.GradScaler()
    D_scaler = torch.cuda.amp.GradScaler()

    step = 0

    for epoch in range(config.NUM_EPOCHS):
        print("Number of Epochs: ", epoch + 1)
        train_fn(G, D, G_optim, D_optim, G_scaler, D_scaler, train_loader, L1_LOSS, BCE, G_logger, D_logger, step)
        if config.SAVE_MODEL and epoch % 5 == 0:
            utils.save_checkpoint(G, G_optim, filename=config.CHECKPOINT_GEN)
            utils.save_checkpoint(D, D_optim, filename=config.CHECKPOINT_DISC)
        if (epoch + 1) % 5 == 0:
            utils.save_results(G, val_loader, epoch)

if __name__ == "__main__":
    main()
