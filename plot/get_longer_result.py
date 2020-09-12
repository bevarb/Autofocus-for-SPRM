'''不同图片之间的亮度会有差异，绘制线性曲线'''
from torchvision.utils import save_image, make_grid
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from models.cycle_models import GeneratorResNet
from models.pix_pix_models import *
from libs.plot_test import plot_test

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import PIL.Image as Image


def de_normolize(image):
    image = (image + 1) / 2
    a = image * 65535
    a = a.cpu().detach().numpy()
    print(a.shape)
    a = np.array(a).transpose((0, 2, 3, 1))
    a = a.astype(np.uint16)
    return a

def save_results(image, weight, height):
    img = np.zeros([height, weight, 3], dtype=np.uint16)
    step = weight // 3
    for i in range(3):
        new = cv.resize(image[i], (height+10, height))
        img[:, i*step:(i+1)*step, :] = new[:, int((height - step) / 2) + 5:int((height - step) / 2) + step + 5]
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray = gray.astype(np.uint16)
    return gray

def change_size(image, size):
    image = np.array(image)
    new = np.zeros([size, size], dtype=image.dtype)  # 新图像

    longer = max(image.shape)
    longer_index = 0 if image.shape[0] == longer else 1
    smaller_index = 1 if longer_index == 0 else 0
    scale = size / longer
    small = int(image.shape[smaller_index] * scale)  # 更短的一边进行放缩
    if longer_index == 0:
        temp = cv.resize(image, (small, size))
        new[:, int((size - small) / 2):int((size - small) / 2) + small] = temp
    else:
        temp = cv.resize(image, (size, small))
        new[int((size - small) / 2):int((size - small) / 2) + small, :] = temp

    return new

def to_rgb(image):
    rgb_image = cv.cvtColor(np.array(image), cv.COLOR_GRAY2RGB)  # 先转为RGB type为uint16

    return rgb_image

def to_tensor(image):
    rgb_image = np.array(image).astype(np.float32)  # 再转为int16 有符号数
    image = rgb_image.transpose((2, 0, 1))
    image = torch.tensor(image).float() / 65535
    image = (image - 0.5) / 0.5
    image[image < -1] = -1
    image[image > 1] = 1
    return image

def SaveResult(real_A, fake_A, flag):
    real_A = (make_grid(real_A, nrow=3, padding=0, normalize=False) + 1) / 2
    fake_A = (make_grid(fake_A, nrow=3, padding=0, normalize=False) + 1) / 2
    error1 = make_grid(torch.sub(real_A, fake_A.type(torch.Tensor)/ real_A), nrow=1, normalize=False)
    image_grid = torch.cat((real_A.type(torch.Tensor), fake_A.type(torch.Tensor), error1.type(torch.Tensor)), 1)
    save_image(image_grid, "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/YaPian/results/%s" % (flag),
               normalize=False)



def pretreat(img):
    mean = np.mean(img)
    img = img.astype(np.int32)
    img = img - mean
    min, max = np.min(img), np.max(img)
    scale = 65535 / (max - min)
    img = img * scale
    img[img[:, :] < 0] = -1 * np.power(np.abs(img[img[:, :] < 0]) / 32500, 0.8) * 32500
    img[img[:, :] > 0] = np.power(np.abs(img[img[:, :] > 0]) / 32500, 0.8) * 32500
    img = img + 32500
    img = img.astype(np.uint16)
    return img




def GetResult(root, name, model_path, size, device_id, kernel_x=5, kernel_y=100):
    cuda = torch.cuda.is_available()
    # Model
    Generator = GeneratorResNet((3, size, size), 25)
    if cuda:
        torch.cuda.set_device(device_id)  # 设置默认使用显卡
        Generator = Generator.cuda()
    # 读入模型

    Generator.load_state_dict(torch.load(model_path))
    Generator.eval()
    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    Image_path = root + name
    Img = np.array(Image.open(Image_path))  # 读入图片
    hight, weight = Img.shape[0], Img.shape[1]
    step = weight // 3
    Input = torch.zeros([3, 3, size, size], dtype=torch.float)
    for i in range(3):
        son = Img[:, i * step:(i + 1) * step]
        son = pretreat(son)
        new = change_size(son, size)
        Input[i, :, :, :] = to_tensor(to_rgb(new))

    pretreat_img = pretreat(Img).astype(np.uint16)

    with torch.no_grad():
        fake_A = Generator(Input.type(Tensor))
        fake_A_ = de_normolize(fake_A)


        fake_result = save_results(fake_A_, weight, hight)

        cv.imwrite("/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/YaPian/img_for_paper/Fake/%s" % name, fake_result)
        cv.imwrite("/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/YaPian/img_for_paper/Real/%s" % name, pretreat_img)
        SaveResult(Input, fake_A, name)




root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/YaPian/clear4/"
name = "867.tif"

GetResult(root, name, '/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/checkpoints/New-Total-Train(move3)_3/G_BA_99.pth',
                   size=256, device_id=1)  # 获取模型结果







