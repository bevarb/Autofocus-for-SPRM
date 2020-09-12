import os
import pandas as pd
import numpy as np
import cv2 as cv
import PIL.Image as Image

def add_max_data(info_root, img_root, if_sub_1):
    '''流程，读取info，
    对每行数据获取对应文件里的info
    读取max值
    '''
    info = pd.read_csv(info_root + "/info.csv")
    data = []
    for i in range(len(info)):
        img_name = info.loc[i]["img_name"]
        img_source = info.loc[i]["source"]
        img_root_name = img_root + img_source.split("/")[0] + "/for_train"
        img_dirname = img_source.split("/")[-3]
        img_sondirname = str(int(img_source.split("/")[-2]) - if_sub_1)  # -1是因为现在的数据源文件名称都减1了
        img_source_name = img_source.split("/")[-1]
        img_info_path = "/".join([img_root_name, img_dirname, img_sondirname, "info.csv"])
        img_info = pd.read_csv(img_info_path)
        img_info = img_info.set_index(["img_name"])
        max_ = img_info.loc[img_source_name]["max"]
        data.append([img_name, max_])
        print(img_name)
    data = pd.DataFrame(data, columns=["img_name", "max"])
    data = data.set_index("img_name")
    info = info.set_index("img_name")
    new = info.join(data, on="img_name")
    new.to_csv(info_root + "/addmaxinfo.csv")
# info_root = "for_total_test/for_show"
# img_root = "/home/user/wangxu_data/Data/z-step_WangXu/New_SubPre/"
# add_max_data(info_root, img_root, if_sub_1=0)

def move_intensity(for_train_root, save_root, info_path, min_max_root):
    '''输入要转换的图片root，然后要保存的root, 以及图片信息info，还有实验图片来源信息
        输出新的图片
    '''
    dir_name = ["Clear_Data_2", "Clear_Data_3", "Clear_Data_5", "Clear_Data_5_T2", "Clear_Data_7", "Clear_Data_6_0.1"]
    min_max = {}
    info = pd.read_csv(info_path)
    info = info.set_index(["img_name"])
    for dir in dir_name:
        min_max_path = min_max_root + "/" + dir + "/for_show/min_max.csv"
        min_max_info = pd.read_csv(min_max_path)
        min_max_info = min_max_info.set_index(["dir_name"])
        min_max[dir] = min_max_info
    for type in ["A", "B"]:
        save_dir = save_root + "/" + type
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        input_dir = for_train_root + "/" + type
        img_names = os.listdir(input_dir)
        for name in img_names:
            img_path = input_dir + "/" + name
            save_path = save_dir + "/" + name
            # 信息获取
            img_min = info.loc[name]["min"]
            # img_max = info.loc[name]["max"]
            if_bit_not = info.loc[name]["if_bit_not"]
            img_source_dir = info.loc[name]["source"].split("/")[2]
            img_source_root = info.loc[name]["source"].split("/")[0]
            dir_min = min_max[img_source_root].loc[img_source_dir]["min"]
            dir_max = min_max[img_source_root].loc[img_source_dir]["max"]
            # 图片处理
            img = np.array(Image.open(img_path))
            if if_bit_not == 1:
                img = cv.bitwise_not(img)
            # cv.imshow("1", img)
            # cv.waitKey(0)
            scale = (dir_max + abs(dir_min)) / 65535  # 这是图像整个返回相当于65535的一个比值

            dir_min, dir_max = dir_min / scale, dir_max / scale
            print(scale, dir_min, dir_max)
            img = img + img_min  # 回到初始相减后的状态
            print(np.min(img), np.max(img))
            img = -1 * img
            # 伽马变换
            img = img.astype(np.int32)
            # 先将负值按照所有图片的最小值进行放缩
            img[img[:, :] < 0] = img[img[:, :] < 0] / scale
            # 再进行变换
            img[img[:, :] < 0] = -1 * np.power(np.abs(img[img[:, :] < 0]) / abs(dir_min), 0.4) * abs(dir_min)
            # 正值
            img[img[:, :] > 0] = img[img[:, :] > 0] / scale
            img[img[:, :] > 0] = np.power(img[img[:, :] > 0] / abs(dir_max), 0.4) * abs(dir_max)

            img = img + abs(dir_min)
            # img = img / ((dir_max + abs(dir_min)) / 65535)
            print(dir_min, np.max(img), np.min(img))
            img[img[:, :] < 0] = 0
            img[img[:, :] > 65535] = 65530
            print(dir_min, np.max(img), np.min(img))
            img = img.astype(np.uint16)

            cv.imwrite(save_path, img)
            print(save_path)
for_train_root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_ROI_Paper/for_train"
save_root = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/test"
info_path = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_ROI_Paper/for_show/info.csv"
min_max_root = "/home/user/wangxu_data/Data/z-step_WangXu/New_SubPre"
move_intensity(for_train_root, save_root, info_path, min_max_root)