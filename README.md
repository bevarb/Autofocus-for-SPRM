# Autofocus-For-SPRM
Use pix2pix-gan solve SPRM defocus problems, and rewrite pix2pix-gan code for study

And Paper address is [paper](wait for add)
### 1 - Introduction
Our code is adapted from [cyclegan](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/cyclegan)
and mainly use the perfect framework in the code (generator and discriminator).

演示
如何测试
If you have train model and get an ".pth", you can use [get_result](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/plot/get_result.py) get autofocus result
![defouc](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/img_for_README/roi_defoucs.png)
![autofoucs](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/img_for_README/roi_foucs.png)
![defocus](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/img_for_README/total_defoucs.png)
![autofocus](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/img_for_README/total_foucs.png)

### 2 - Train

##### First-Prepare your data
We mainly have two dataset, A and B. The A is target of B. We use the name of the picture to bind A and B.

A: 1-17

B: 1-11、1-12...1-45

The number on the "-" left is the serial number, representing a different sequence.And the number on the
"-" right is the id, representing a different focus image. So 1-XX -> 1-17, according to this relationship, 
the binding relationship of other sequences can be established.

##### Second - Prepare two info.csv

1：You can use [get_csv.py](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/data/get_csv.py) get data.csv, Due to the difference in the number of pictures at different focal lengths, 
csv is used for statistics, which can facilitate the subsequent segmentation of the training set and the verification set,
 and the selection of the focal length range.

2：When getting pictures, the information of each picture is stored in info.csv, which is convenient for data preprocessing.
 If you don’t need this, you can modify [DataLoader.py](https://github.com/bevarb/Autofocus-for-SPRM/DataLoader.py)
 
##### Third - Train and View training samples regularly
Before training, you can modify various parameters: picture size, number of channels, learning rate, number of residual blocks. . .

Use [ROI_train.py](https://github.com/bevarb/Autofocus-for-SPRM/ROI_train.py) or [Total_train.py](https://github.com/bevarb/Autofocus-for-SPRM/Total_train.py) for training, and every 20(you can modify) batch, output samples to
images/Sample Name/x.tif

You can stop training when the sample becomes stable.

### 3 - Tools

3.1 [get_clear_z_step.py](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/data/get_clear_z_step.py) and [get_z_step_train_data.py](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/data/get_z_step_train_data.py)

 After getting the original SPRM data, you can get the subtracted image through the first code, 
 and put them into two folders: for_show、for_train. This is because the intensity value of the picture is too low, it is
  difficult to filter the training set, you can view the zoomed picture in the for_show folder
  
Then use the second code to select the sequence and range of SPRM pictures in the data set and put them into the training set,
 and you can choose whether to segment the image.
 
3.2 [pre_treat.py](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/data/pre_treat.py)

You can use this .py to pretreat your dataset and when you train you don't need pretreat.

3.3 [delete_move.py](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/data/delete_move.py)

Delete some bad pictures in folder "for_show" or select the target pictures in it, and the same operation should be done 
in folder "for_train"". You can use this code

3.4 [Autofocus used tiled sensor chip SPRM img](https://github.com/bevarb/Autofocus-for-SPRM/blob/master/plot/get_longer_result.py)

If you have a longer image, you can use this code to split the image to focus
  




