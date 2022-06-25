import torch
import numpy as np
import matplotlib.pyplot as plt
import config
from skimage import color

def save_results(G, val_loader, epoch, folder=config.SAVE_DIR):
    data = next(iter(val_loader))
    Ls, ABs = data['L'], data['ab']
    Ls = Ls.to(config.DEVICE)
    ABs = ABs.to(config.DEVICE)
    G.eval()
    with torch.no_grad():
        fake_color = G(Ls)
        fake_imgs = lab_to_rgb(Ls, fake_color)
        real_imgs = lab_to_rgb(Ls, ABs)
        fig = plt.figure(figsize=(15, 9))
        for i in range(5):
            ax = plt.subplot(3, 5, i + 1)
            ax.imshow(Ls[i][0].cpu(), cmap='gray')
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 5)
            ax.imshow(fake_imgs[i])
            ax.axis("off")
            ax = plt.subplot(3, 5, i + 1 + 10)
            ax.imshow(real_imgs[i])
            ax.axis("off")
        plt.show()
        fig.savefig(f"{folder}val_result_{epoch}.png")
    G.train()

def test_an_image(G, test_img_loader, img_name="", folder=config.SAVE_DIR):
    data = next(iter(test_img_loader))
    Ls = data['L'].to(config.DEVICE)
    G.eval()
    with torch.no_grad():
        fake_color = G(Ls)
        fake_imgs = lab_to_rgb(Ls, fake_color)
        fig = plt.figure(figsize=(6, 12))
        ax = plt.subplot(2, 1, 1)
        ax.imshow(Ls[0][0].cpu(), cmap='gray')
        ax.axis("off")
        ax = plt.subplot(2, 1, 2)
        ax.imshow(fake_imgs[0])
        ax.axis("off")
        plt.show()
        fig.savefig(f"{folder}test_result_{img_name}.png")

def save_checkpoint(model, optimizer, filename="/my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optim_state"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(-1, 1)

def lab_to_rgb(L, ab):
    L, ab = denorm(L), denorm(ab)
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        rgb_img = color.lab2rgb(img)
        rgb_imgs.append(rgb_img)
    return np.stack(rgb_imgs, axis=0)
