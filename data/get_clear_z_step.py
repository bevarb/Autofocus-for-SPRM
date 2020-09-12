import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
'''本代码将不同焦距的原始图像，减去前一帧，得到clear图像
   方法是每次实验一个文件夹中对应的图片，减去上一个序列中对应的图片
'''

def change_name(root, son_root):
    '''处理数据出现顺序反过来的情况，要将名字反过来'''
    for k in ["show", "train"]:
        dir = os.listdir(root + "/for_%s/" % k + son_root)
        dir = [root + "/for_%s/" % k + son_root + "/" + d for d in dir]
        flag = ["_.", "."]
        for i in range(2):
            for d in dir:
                path = os.listdir(d)
                l = len(path) - 1  # 有一个是数据文件
                for p in path:
                    if ".tif" in p:
                        old = d + "/" + p
                        s = int(p.split(flag[1 - i])[0])
                        if flag[i] == "_.":
                            # 第一次循环将名字顺序反过来，而且加一个"_"
                            new = d + "/" + str(l - s + 1) + flag[i] + "tif"
                            os.rename(old, new)
                            print(old, new)
                        else:
                            # 第二次只将名字的"_"去掉
                            new = d + "/" + str(s) + flag[i] + "tif"
                            os.rename(old, new)
                            print(old, new)




def change_info(root, son_root):
    '''处理数据出现顺序反过来的情况，要将名字反过来'''
    for k in ["show", "train"]:
        dir = os.listdir(root + "/for_%s/" % k + son_root)
        dir = [root + "/for_%s/" % k + son_root + "/" + d for d in dir]

        for d in dir:
            path = os.listdir(d)
            l = len(path) - 1  # 有一个是数据文件 61个图像就应该为61
            info = pd.read_csv(d + "/info.csv")
            new_info = []  # 存放新的信息文件
            for i in range(1, l+1):
                src = l - i + 1  # 倒着来，从61开始-1
                temp = [str(l - src + 1) + ".tif", info.iloc[src - 1]["min"], info.iloc[src - 1]["max"], info.iloc[src - 1]["if_bit_not"]]
                new_info.append(temp)

            new_info = pd.DataFrame(new_info, columns=["img_name", "min", "max", "if_bit_not"])

            # os.remove(d + "/info.csv")
            # os.rename(d + "/new_info.csv", d + "/info.csv")

            new_info.to_csv(d + "/info.csv", index=False)
            print("have_update_info %s" % d + "/info.csv")

# change_name("/home/user/wangxu_data/Data/z-step_WangXu/New_SubPre/Clear_Data_5", "2800-2812-0.2")
# change_info("/home/user/wangxu_data/Data/z-step_WangXu/New_SubPre/Clear_Data_5", "2800-2812-0.2")


def sub_image(root, save_root, method="Sub Pri", bit_not=True, T=2):
    dir_name = os.listdir(root)
    for j in range(len(dir_name)):
        dir = root + "/" + dir_name[j]  # 不同批次的实验
        for d in ["show", "train"]:  # 不同的处理效果
            print(d, dir_name[j])
            save_dir = save_root + "/for_%s/" % d + dir_name[j]  # 保存的root位置
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            son_dir_list = os.listdir(dir)  # 单次实验不同文件夹
            son_dir_list = sorted(son_dir_list, key=lambda x: int(x))
            for k in range(len(son_dir_list) - 1):  # 同批次中不同的子文件夹

                if method == "Sub Pri":
                    First = dir + "/" + son_dir_list[k]
                else:
                    First = dir + "/" + son_dir_list[(k // T) * T]
                son_dir = dir + "/" + son_dir_list[k+1]
                son_save_dir = save_dir + "/" + str(int(son_dir_list[k+1])-1)
                print(First, son_save_dir)
                if not os.path.exists(son_save_dir):
                    os.makedirs(son_save_dir)

                First_name = os.listdir(First)
                First_name = sorted(First_name, key=lambda x: int(x.split("Z")[-1].split(".")[0]))

                source_name = os.listdir(son_dir)
                source_name = sorted(source_name, key=lambda x: int(x.split("Z")[-1].split(".")[0]))

                First_path = [First + "/" + name for name in First_name]
                source_path = [son_dir + "/" + name for name in source_name]
                info = []
                for i in range(min(len(source_path), len(First_path))):  # 单个文件夹的每个图片
                    if os.path.exists(First_path[i]) and os.path.exists(source_path[i]):
                        # img1 = cv2.imread(First_path[i])
                        # img2 = cv2.imread(source_path[i])
                        img1 = np.array(Image.open(First_path[i])).astype(np.int32)
                        img2 = np.array(Image.open(source_path[i])).astype(np.int32)
                        img = img2 - img1
                        if bit_not == True:
                            img = -1 * img
                        min_ = np.min(img)
                        max_ = np.max(img)
                        #img = np.abs(img)
                        if d == "show":
                            img = img / 0.5
                            img = img + np.abs(np.min(img))
                        else:
                            img = img + np.abs(np.min(img))
                        img = img.astype(np.uint16)
                        # if d != "show":
                        #     print(img.min(), img.max())

                        info.append(["%d.tif" % (i + 1), min_, max_, 1 if bit_not else 0])
                        cv2.imwrite(son_save_dir + "/%d.tif" % (i + 1), img)
                info = pd.DataFrame(info, columns=["img_name", "min", "max", "if_bit_not"])
                info.to_csv(son_save_dir + "/info.csv", index=False)

# root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/YaPian/raw"
# save_root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/YaPian/clear"

root = "/home/user/wangxu_data/Data/z-step_WangXu/Raw_Data_7"
save_root = "/home/user/wangxu_data/Data/z-step_WangXu/New_SubPre/Clear_Data_7"
sub_image(root, save_root, method="Sub Pri", bit_not=True, T=5)

