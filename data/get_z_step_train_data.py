import os
import PIL.Image as Image
import numpy as np
import cv2 as cv
import pandas as pd
def get_z_step_train(dir, save_dir, begain, over, bbox=[], spilt=True):

    os.makedirs(save_dir, exist_ok=True)
    info_save_dir = save_dir[0:len(save_dir)-1] + "info.csv"  # info要保存的位置

    type = 0
    # 以下是得到对应文件里有多少个ID了
    if len(os.listdir(save_dir)) == 0:
        type = 1
    else:
        new_name = os.listdir(save_dir)
        new_name = sorted(new_name, key=lambda x: int(x.split("_")[0]))
        type = int(new_name[-1].split("_")[0]) + 1
    # 以下是得到图像
    temp_info = []
    for i in range(begain, over + 1):
        name = str(i) + ".tif"
        path = dir + "/" + name
        img = np.array(Image.open(path))
        img_info = pd.read_csv(dir + "/info.csv")
        img_info = img_info.set_index("img_name")

        temp = ["%d_%d.tif" % (type, i), img_info.loc[str(i) + ".tif"]["min"], img_info.loc[str(i) + ".tif"]["if_bit_not"]]
        source = "/".join(dir.split("/")[7:]) + "/" + name
        temp.append(source)
        if spilt == True:
            img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            temp.append(bbox)
        cv.imwrite(save_dir + "/" + ("%d_%d.tif" % (type, i)), img.astype(np.uint16))
        temp_info.append(temp)
    if spilt:
        temp_info = pd.DataFrame(temp_info, columns=["img_name", "min", "if_bit_not", "source", "ROI"])
    else:
        temp_info = pd.DataFrame(temp_info, columns=["img_name", "min", "if_bit_not", "source"])
    if os.path.exists(info_save_dir):
        old_info = pd.read_csv(info_save_dir)
        new = pd.concat([old_info, temp_info])
        new.to_csv(info_save_dir, index=False)
        print(temp_info)
        print("have concat new_info to old_info")
    else:
        temp_info.to_csv(info_save_dir, index=False)
        print("have new info.csv and info:")
        print(temp_info)

        # cropped = img((bbox[0], bbox[1], bbox[2], bbox[3]))
        # cropped.save(save_dir + "/" + ("%d_%d.tif" % (type, i)))

# 以下是处理整幅图像
for s in ["for_show", "for_train"]:
    dir = "/home/user/wangxu_data/Data/z-step_WangXu/New_SubPre/Clear_Data_7/%s/" \
          "2803-2813-2-1/19" % s
    save_dir = "/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_total_test" \
               "/%s/B" % s
    bbox1 = [96, 1, 432, 345]
    get_z_step_train(dir, save_dir, 1, 48, bbox=bbox1, spilt=False)

# li = [2, 3, 4, 5, 6, 7, 8]
# for l in li:
#     dir = "/home/user/wangxu_data/Data/z-step_WangXu/Clear_Data_4_0.1/Sub First/2834-2843-0.1-1油太多/%d" % l
#     save_dir = "/home/user/wangxu_data/code/2-AutoDetect/Gan_App_pytorch/data/ROI-Z-step-0.1/B"
#     bbox1 = [328, 1, 595, 272]
#     get_z_step_train(dir, save_dir, 1, 63, bbox=bbox1, spilt=True)

# dir = "/home/user/wangxu_data/Data/z-step_WangXu/Clear_Data_5/2814-2827-0.2/63"
# save_dir = "/home/user/wangxu_data/Data/备份/ROI-Z-step-0.2/B"
# bbox1 = [217, 1, 485, 322]
# get_z_step_train(dir, save_dir, 18, 61, bbox=bbox1, spilt=True)

