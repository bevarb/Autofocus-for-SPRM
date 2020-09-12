import glob
import random
import pandas as pd
import os
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms as transforms
import torch



def to_rgb(image):
    rgb_image = cv.cvtColor(np.array(image), cv.COLOR_GRAY2RGB)  # 先转为RGB type为uint16
    rgb_image = np.array(rgb_image).astype(np.float32)  # 再转为int16 有符号数
    rgb_image = rgb_image.transpose((2, 0, 1))
    return rgb_image
def to_tenser(image):
    return torch.tensor(image).float() / 65535

def normolize_(image):
    rgb_image = (image - 0.5) / 0.5
    rgb_image[rgb_image < -1] = -1
    rgb_image[rgb_image > 1] = 1

    return rgb_image

def change_size(image, size):
    image = np.array(image).astype(np.uint16)
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

    # new = np.zeros([longer, longer], dtype=image.dtype)
    # begain_x, begain_y = (longer - image.shape[0]) // 2, (longer - image.shape[1]) // 2
    # new[begain_x:begain_x + image.shape[0], begain_y:begain_y + image.shape[1]] = image
    # new = cv.resize(new, (size, size))
    return new


def split_TrainTest(csvpath, begain, over, trainsize=0.85):
    data = pd.read_csv(csvpath)
    data = data.loc[:, begain:over]
    train = []
    test = []
    testData = [["" for _ in range(int(begain), int(over))] for _ in range(200)]
    for j in range(int(begain), int(over)):
        if j != 0:
            current, current2 = 0, 0
            train_nums = int(int(data.loc[len(data) - 1, str(j)]) * trainsize)
            for i in range(len(data) - 1):
                if len(str(data.loc[i, str(j)])) > 5:
                    if current < train_nums:
                        train.append(data.loc[i, str(j)])
                        current += 1
                    else:
                        test.append(data.loc[i, str(j)])
                        testData[current2][j + abs(int(begain))] = (data.loc[i, str(j)])
                        current2 += 1
    testData = pd.DataFrame(testData, columns=[str(i) for i in range(int(begain), int(over), 1)])
    for i in range(len(testData)):
        if testData.loc[i, "1"] != "":
            testData.loc[i, "0"] = str(testData.loc[i, "1"]).split("_")[0] + "_" + \
                               str(int(str(testData.loc[i, "1"]).split("_")[-1].split(".")[0]) -1) + ".tif"
    testData.to_csv("./data/test.csv", index=False)
    return train, test

class TrainDataset(Dataset):
    def __init__(self, dataset, root, size, channels, config, flag):
        self.size = size
        self.data = dataset
        self.root = root
        self.channels = channels
        self.info = pd.read_csv(root + "/info.csv")
        self.info = self.info.set_index("img_name")
        fileAlist = os.listdir(root + "/A/")
        self.files_A = sorted(fileAlist, key=lambda x: int(x.split("_")[0]))
        self.flag = flag
        self.config = config

    def __getitem__(self, index):

        image_B = Image.open(self.root + "/B/" + self.data[index])
        A_index = int(self.data[index].split("/")[-1].split("_")[0]) - 1

        image_A = Image.open(self.root + "/A/" + self.files_A[A_index])
        A_name, B_name = self.data[index], self.files_A[A_index]
        A_min, B_min = self.info.loc[A_name]["min"], self.info.loc[B_name]["min"]

        scale = 1.
        min_scale = 1.
        if self.flag != "abs":
            image_A, image_B = self.transfor_normal_data(image_A, image_B, A_name, B_name, A_min, B_min)



        return {"A": image_A, "B": image_B, "scale":scale, "min_scale":min_scale}

    def __len__(self):
        return len(self.data)

    def transfor_normal_data(self, image_A, image_B, A_name, B_name, A_min, B_min):
        image_A, image_B = np.array(image_A), np.array(image_B)
        # 如果if_bit_not == 1，说明之前进行过取补数，故再进行取补回到原来
        # if self.info.loc[A_name]["if_bit_not"] == 1:
        #     image_A = cv.bitwise_not(np.array(image_A))
        #     image_B = cv.bitwise_not(np.array(image_B))
        # # 加上最小值，回到0中心
        # image_A = image_A + A_min
        # image_B = image_B + B_min


        # 获取信息
        min_A, min_B = np.min(image_A), np.min(image_B)
        max_A, max_B = np.max(image_A), np.max(image_B)
        mean_A, mean_B = np.mean(image_A), np.mean(image_B)
        min_ = np.min([min_A, min_B])
        max_ = np.max([max_A, max_B])
        max = np.max([abs(min_A), abs(min_B), max_A, max_B])
        # print([abs(min_A), abs(min_B), max_A, max_B])
        # scale = (max_ - min_) / (65535)
        scale = 0.5
        max_, min_ = max_ / scale, min_ / scale
        # print([abs(min_A), abs(min_B), max_A, max_B], scale)

        image_A, image_B = image_A - mean_A, image_B - mean_B

        # 如果if_bit_not == 1，还说明该粒子最大值在负数那里，需要反过来
        if self.info.loc[A_name]["if_bit_not"] == 1:
            image_A = image_A * (-1)
            image_B = image_B * (-1)

        # 放缩
        image_A = image_A.astype(np.int32) / scale
        image_B = image_B.astype(np.int32) / scale

        # gamma
        image_A, image_B = image_A.astype(np.float32), image_B.astype(np.float32)
        g_min, g_max = self.config["use_gama"][0], self.config["use_gama"][1]
        g = round(random.uniform(g_min, g_max), 2)
        score = 32500

        image_A[image_A[:, :] < 0] = (-1) * np.power(np.abs(image_A[image_A[:, :] < 0]) / score, g) * score
        image_A[image_A[:, :] > 0] = np.power(np.abs(image_A[image_A[:, :] > 0]) / score, g) * score

        image_B[image_B[:, :] < 0] = (-1) * np.power(np.abs(image_B[image_B[:, :] < 0]) / score, g) * score
        image_B[image_B[:, :] > 0] = np.power(np.abs(image_B[image_B[:, :] > 0]) / score, g) * score

        image_A[image_A < -score] = -score
        image_A[image_A > score] = score
        image_B[image_B < -score] = -score
        image_B[image_B > score] = score

        # 获取基本信息
        image_A = image_A + score
        image_B = image_B + score

        if self.config["use_flip"] == True:
            bit_flip = np.random.rand() < .5
            if bit_flip:  # 随机翻转
                image_A = Image.fromarray(image_A).transpose(Image.FLIP_LEFT_RIGHT)
                image_B = Image.fromarray(image_B).transpose(Image.FLIP_LEFT_RIGHT)
                image_A, image_B = np.array(image_A), np.array(image_B)

        # bit_not = np.random.rand() < .2
        # if bit_not:  # 随机像素取补
        #     image_A = cv.bitwise_not(np.array(image_A))
        #     image_B = cv.bitwise_not(np.array(image_B))
        # min = np.min(image_A) / 65535 if np.min(image_A) < np.min(image_B) else np.min(image_B) / 65535

        # 调整大小
        image_A = change_size(np.array(image_A), self.size)
        image_B = change_size(np.array(image_B), self.size)

        # 调为三通道， 返回的是numpy
        image_A = to_rgb(image_A)
        image_B = to_rgb(image_B)
        # 转为tensor
        image_A = to_tenser(image_A)
        image_B = to_tenser(image_B)

        if self.config["use_random_add_intensity"] == True:
            # 随机调整亮度
            if_scale = np.random.randint(-5, 5) / 10
            if if_scale > 0.3:
                scale = random.uniform(-0.2, 0.2)
                image_A = image_A + scale
                image_B = image_B + scale

        image_A = normolize_(image_A)
        image_B = normolize_(image_B)


        return image_A, image_B

class yapianDataset(Dataset):
    def __init__(self, root, size, begain, over):
        self.size = size

        self.root = root
        names = os.listdir(root)
        names = sorted(names, key=lambda x: int(x.split("_")[1]))
        self.begain = begain
        self.over = over
        self.paths = [root + "/" + names[i] for i in range(len(names))]


    def __getitem__(self, index):

        Img1 = np.array(Image.open(self.paths[self.begain + index])).astype(np.int32)  # 读入图片
        Img2 = np.array(Image.open(self.paths[self.begain + index + 10])).astype(np.int32)
        Img = Img2 - Img1
        hight, weight = Img.shape[0], Img.shape[1]
        step = weight // 3
        sons = []
        for i in range(3):
            son = Img[:, i * step:(i + 1) * step]
            sons.append(son)
        ran = np.random.randint(0, 2)
        image_A = sons[ran]
        image_B = sons[ran]

        image_A, image_B = np.array(image_A), np.array(image_B)


        # 获取信息
        min_A, min_B = np.min(image_A), np.min(image_B)
        max_A, max_B = np.max(image_A), np.max(image_B)
        mean_A, mean_B = np.mean(image_A), np.mean(image_B)

        scale = 0.5
        image_A, image_B = image_A - mean_A, image_B - mean_B

        # 放缩
        image_A = image_A.astype(np.int32) / scale
        image_B = image_B.astype(np.int32) / scale

        # gamma
        image_A, image_B = image_A.astype(np.float32), image_B.astype(np.float32)
        g = round(random.uniform(0.4, 0.8), 2)
        score = 32500

        image_A[image_A[:, :] < 0] = (-1) * np.power(np.abs(image_A[image_A[:, :] < 0]) / score, g) * score
        image_A[image_A[:, :] > 0] = np.power(np.abs(image_A[image_A[:, :] > 0]) / score, g) * score

        image_B[image_B[:, :] < 0] = (-1) * np.power(np.abs(image_B[image_B[:, :] < 0]) / score, g) * score
        image_B[image_B[:, :] > 0] = np.power(np.abs(image_B[image_B[:, :] > 0]) / score, g) * score

        image_A[image_A < -score] = -score
        image_A[image_A > score] = score
        image_B[image_B < -score] = -score
        image_B[image_B > score] = score

        # 获取基本信息
        image_A = image_A + score
        image_B = image_B + score

        # 调整大小
        image_A = change_size(np.array(image_A), self.size)
        image_B = change_size(np.array(image_B), self.size)

        # 调为三通道， 返回的是numpy
        image_A = to_rgb(image_A)
        image_B = to_rgb(image_B)
        # 转为tensor
        image_A = to_tenser(image_A)
        image_B = to_tenser(image_B)

        image_A = normolize_(image_A)
        image_B = normolize_(image_B)



        return {"A": image_A, "B": image_B}

    def __len__(self):
        return self.over - self.begain

