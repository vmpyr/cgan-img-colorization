import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from skimage import color

class ImgDataset(Dataset):
    def __init__(self, img_dir, is_train=True, is_test=False):
        self.is_test = is_test
        self.list_files = []
        if is_test: self.list_files.append(img_dir)
        else:
            self.img_dir = img_dir
            self.list_files = os.listdir(self.img_dir)
            np.random.seed(50)
            self.list_files = np.random.choice(self.list_files, 10_000, replace=False)  # choosing 10000 images randomly
            if is_train: self.list_files=self.list_files[:8000]
            else: self.list_files=self.list_files[8000:]

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        if self.is_test:
            img_path = self.list_files[idx]
        else:
            filename = self.list_files[idx]
            img_path = os.path.join(self.img_dir, filename)
        img = Image.open(img_path)
        width, height = img.size
        if width > height: img.thumbnail((width, 256), resample=Image.LANCZOS)
        else: img.thumbnail((256, height), resample=Image.LANCZOS)
        im_temp = img.crop((0, 0, 256, 256))
        im = im_temp.convert('RGB')
        im = transforms.ToTensor()(im)
        im = np.array(im)
        im = np.transpose(im, (1,2,0))
        lab = color.rgb2lab(im).astype(np.float32)
        lab_t = transforms.ToTensor()(lab)
        inp_transform = transforms.Compose([
            transforms.Normalize((0.5), (0.5))
        ])
        targ_transform = transforms.Compose([
            transforms.Normalize((0.5, 0.5), (0.5, 0.5))
        ])
        input = inp_transform((lab_t[[0], ...] / 50.0) - 1.0)
        target = targ_transform(lab_t[[1, 2], ...] / 110.0)

        return {'L': input, 'ab': target}
