import pandas as pd
import numpy as np
import os

def get_csv(B_root, A_root, save_root, CENTER=100):
    '''获得A、B文件夹图片的序列信息，找到对焦帧，存入csv文件'''
    A_List = os.listdir(A_root)
    A_List = sorted(A_List, key=lambda x: int(x.split("_")[0]))
    B_List = os.listdir(B_root)
    B_List = sorted(B_List, key=lambda x: int(x.split("_")[0]))
    Data = [["None" for _ in range(CENTER*2)] for _ in range(len(A_List))]
    for i in range(len(A_List)):
        Data[i][CENTER] = A_List[i]
        id = int(A_List[i].split("_")[0])
        A_frame = int(A_List[i].split("_")[-1].split(".")[0])
        B_path = []
        for B in B_List:
            if int(B.split("_")[0]) == id:
                B_path.append(B)
        B_path = sorted(B_path, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        for B in B_path:
            B_frame = int(B.split("_")[-1].split(".")[0])
            real = A_frame - B_frame  # 实际失焦帧
            Data[i][CENTER - real] = B  # 在对应位置上添加
    nums = []
    # 计算每个焦距存在的图片数量
    for i in range(CENTER*2):
        num = 0
        for j in range(len(A_List)):
            if Data[j][i] != "None":
                num += 1
            elif Data[j][i] == "None":
                Data[j][i] = ""
        nums.append(num)
    Data.append(nums)
    Data = pd.DataFrame(Data, columns=[str(i) for i in range(-100, 100, 1)])
    Data.to_csv(save_root + "/" + "data.csv", index=False)
    print(Data)
def get_test_csv(root):
    '''获得0.1序列下的数据，所有数据都按从左到右的顺序，长短不一'''
    All_name = os.listdir(root + "/B")
    All_name = sorted(All_name, key=lambda x: int(x.split("_")[0]))
    L = int(All_name[-1].split("_")[0])
    All_DATA = []
    for i in range(L):
        temp = ["NONE" for i in range(100)]
        All_DATA.append(temp)
    for name in All_name:
        path = name
        x, y = int(name.split("_")[0]) - 1, int(name.split("_")[-1].split(".")[0]) - 1
        print(path, x, y)
        All_DATA[x][y] = path
    # 记录时间
    t = []
    for i in range(100):
        flag = 0
        for j in range(L):
            if All_DATA[j][i] != "NONE":
                flag += 1
        t.append(flag)
    All_DATA.append(t)
    Data = pd.DataFrame(All_DATA, columns=[str(i) for i in range(0, 100)])
    Data.to_csv(root + "/" + "newdata.csv", index=False)
# get_test_csv("New-ROI-0.1/test1/new_for_train_2")


B_root = "New-Total-Train/for_train/B"
A_root = "New-Total-Train/for_train/A"
save_root = "New-Total-Train/for_train"
get_csv(B_root, A_root, save_root)