'''
现有的数据包含两个,for_show   for_train
所以删除文件不能直接删除一个，而是对应的两个

'''
import os
import shutil
def delete_data(root, flag1="for_show", flag2="for_train"):
    '''对于fow_show里面已经删除的B，应该将for_train中B里对应名称的数据也删除'''
    names = os.listdir(root + "/%s/B" % flag2)
    for_show_path = root + "/%s" % flag1
    for_show_B_names = os.listdir(for_show_path + "/B")
    for name in names:
        if name not in for_show_B_names:
            remove_path = root + "/%s" % flag2 + "/B/%s" % name
            os.remove(remove_path)
            print("have delete %s" % remove_path)


def remove_data(root, flag1="for_show", flag2="for_train"):
    '''对于fow_show里面已经筛选出来的A，应该将for_train中B里对应名称的数据也放入对应的A'''
    names = os.listdir(root + "/%s/A" % flag1)
    for_train_path = root + "/%s" % flag2
    for_train_B_names = os.listdir(for_train_path + "/B")
    for name in names:
        if name in for_train_B_names:
            old_path = for_train_path + "/B/%s" % name
            new_path = for_train_path + "/A/%s" % name
            shutil.move(old_path, new_path)
            print(old_path, new_path)
def remove_test_data(root, flag1="for_show(abs)", flag2="for_train"):
    '''将for_show中的test集在for_train中同样生成一个'''
    for type in ["A", "B"]:
        src_root = root + "/%s" % flag1 + "/test/" + type
        src_names = os.listdir(src_root)
        for name in src_names:
            img_path = root + "/%s/" % flag2 + type + "/" + name
            save_root = root + "/%s" % flag2 + "/test/" + type
            os.makedirs(save_root, exist_ok=True)
            save_path = save_root + "/" + name
            if os.path.exists(img_path):
                shutil.move(img_path, save_path)
                print(img_path, save_path)



# remove_data("/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_Fitted",
#             flag1="for_show", flag2="for_train")
# delete_data("/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/for_Fitted",
#            flag1="for_show", flag2="for_train(abs)")
remove_test_data("/home/user/wangxu_data/code/1_AutoFocus/Gan_App_pytorch/data/New-ROI-Train")
