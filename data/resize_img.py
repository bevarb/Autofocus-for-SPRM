import os
from PIL import Image
import numpy as np
import cv2 as cv
dir = "./A"
savedir = "./B"
size_x = 128
size_y = 128
filename = os.listdir(dir)

for name in filename:
    image = np.array(Image.open(dir + "/" + name))

    image = cv.resize(image, (size_x, size_y), interpolation=cv.INTER_CUBIC)
    print(image)
    cv.imwrite(savedir + "/" + name, image)

    # image = to_rgb(image)
    # new = image.resize((size_x, size_y))
    # new.save(
