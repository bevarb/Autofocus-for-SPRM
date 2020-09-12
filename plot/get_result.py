import os
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
from scipy import optimize
from models.pix_pix_models import *
from torchvision.utils import *
import pandas as pd
import time
class plot():
    def __init__(self, model_path, nums, devices_id, size):
        self.devices_id = devices_id
        self.size = size
        self.init_model(model_path, nums)

    def init_model(self, model_path, nums):
        # 配置模型
        self.cuda = torch.cuda.is_available()
        # Model
        self.Generator = GeneratorResNet((3, self.size, self.size), nums)
        if self.cuda:
            torch.cuda.set_device(self.devices_id)  # 设置默认使用显卡
            self.Generator = self.Generator.cuda()
        self.Generator.load_state_dict(torch.load(model_path))
        self.Generator.eval()
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.Tensor


    def to_rgb(self, image):
        rgb_image = cv.cvtColor(np.array(image), cv.COLOR_GRAY2RGB)  # 先转为RGB type为uint16
        rgb_image = rgb_image.astype(np.float32)  # 再转为int16 有符号数
        rgb_image = rgb_image.transpose((2, 0, 1))
        return rgb_image

    def normolize(self, image):
        rgb_image = torch.tensor(image).float() / 65535
        rgb_image = (rgb_image - 0.5) / 0.5
        rgb_image[rgb_image < -1] = -1
        rgb_image[rgb_image > 1] = 1
        return rgb_image

    def change_size(self, image, size):
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

    def pretreat(self, img, phase_inverse):
        mean = np.mean(img)
        img = img.astype(np.int32)
        img = img - mean
        if phase_inverse:
            img = -1 * img
        min, max = np.min(img), np.max(img)
        scale = 65535 / (max - min)
        img = img * scale
        img[img[:,:]<0] = -1 * np.power(np.abs(img[img[:,:]<0]) / 32500, 0.5) * 32500
        img[img[:, :] > 0] = np.power(np.abs(img[img[:, :] > 0]) / 32500, 0.5) * 32500
        img = img + 32500
        img = img.astype(np.uint16)
        return img


    def de_normorlize(self, img):
        temp = np.array(img * 65535)
        temp = temp.transpose((1, 2, 0))
        ndarr = temp.astype(np.uint16)
        return ndarr

    def show_img(self, img, i, flag):
        def on_Event(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                point = [x, y]
                self.points[i][flag] = point
                print(point)

        cv.namedWindow("%d-%d" % (i, flag))
        cv.setMouseCallback("%d-%d" % (i, flag), on_Event)
        while (1):
            cv.imshow("%d-%d" % (i, flag), img)
            if cv.waitKey(200) & 0xFF == 27:
                break
        cv.destroyAllWindows()

    def split_black_margin(self, image, shape):

        longer = max(shape)
        longer_index = 0 if shape[0] == longer else 1
        shorter_index = 1 if longer_index == 0 else 0
        shorter = shape[shorter_index]
        dx = int((longer - shorter) / 2)
        new = np.zeros(shape, dtype=np.uint16)
        image = cv.resize(image, (longer, longer))
        if longer_index == 0:
            new = image[:, dx:int(dx+shorter), :]
        else:
            new = image[dx:int(dx+shorter), :, :]
        return new


    def main(self, root, save_root, use_pretreat, phase_inverse):

        "1先处理图片，输入图片为不同亮度的" \
        "2输入模型，得到生成图片" \
        "3对两组图片的spot计算，返回两组值"
        names = os.listdir(root)
        for name in names:
            path = root + "/" + name
            start_time = time.time()
            # 打开图片
            img = np.array(Image.open(path))
            shape = img.shape
            # 图片预处理
            if use_pretreat:
                img = self.pretreat(img, phase_inverse)
            # 转换大小并且转三通道
            img = self.to_rgb(self.change_size(img, self.size))
            img = self.normolize(img)
            img1 = (img + 1) / 2  # 原来为-1～1， 转为0～1
            img2 = torch.unsqueeze(img, dim=0)  # 增加维度
            if self.cuda:
                img2 = img2.type(self.Tensor)
            img2 = (self.Generator(img2) + 1) / 2
            img2 = img2.cpu().detach().numpy()
            img2 = img2[0]
            # 转为0～1
            img1_ = self.de_normorlize(img1)
            img2_ = self.de_normorlize(img2)
            # img2__ = self.split_black_margin(img2_, shape)
            over_time = time.time()
            print(over_time-start_time)
            os.makedirs(save_root + "/Real", exist_ok=True)
            os.makedirs(save_root + "/Fake", exist_ok=True)

            cv.imwrite(save_root + "/Real/" + name, img1_)
            cv.imwrite(save_root + "/Fake/" + name, img2_)




# # 单个粒子
# root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_ROI_Paper/for_train/B"  # 图片根目录
# save_root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_ROI_Paper/Results_for_paper"
# model_path = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/checkpoints/New-ROI-Train(move1)/G_BA_89.pth"
# plot = plot(model_path, 25, 3, 256)
# plot.main(root, save_root, use_pretreat=True, phase_inverse=True)
# 多个粒子
root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_total_test/for_train/B"  # 图片根目录
save_root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_total_test/Results_for_paper"
model_path = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/checkpoints/New-Total-Train(move3)_2/G_BA_99.pth"
plot = plot(model_path, 25, 3, 256)
plot.main(root, save_root, use_pretreat=True, phase_inverse=False)



