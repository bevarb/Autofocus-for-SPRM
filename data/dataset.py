import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class ImageDataset(Dataset):
    def __init__(self, A_root, B_root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        name_A = os.listdir(A_root)
        name_B = os.listdir(B_root)
        self.name_A = sorted(name_A, key=lambda x: int(x.split("_")[0]))
        self.files_A = [A_root + "/" + name for name in self.name_A]
        self.files_B = [B_root + "/" + name for name in name_B]


    def __getitem__(self, index):
        image_B = Image.open(self.files_B[index])
        A_index = int(self.files_B[index].split("/")[-1].split("_")[0]) - 1

        image_A = Image.open(self.files_A[A_index])

        # Convert grayscale images to rgb
        if image_A.mode != "RGB":
            image_A = to_rgb(image_A)
        if image_B.mode != "RGB":
            image_B = to_rgb(image_B)

        item_A = self.transform(image_A)
        item_B = self.transform(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return len(self.files_B)