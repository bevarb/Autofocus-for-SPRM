import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

#from models.cycle_models import *
from models.models_for_total import *
from total_dataloader import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="New-Total-Train(move3)_test", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--lr_G", type=float, default=0.00025, help="adam: learning rate")
parser.add_argument("--lr_D", type=float, default=0.000007, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=5, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=512, help="size of image height")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--checkpoint_interval", type=int,  default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_residual_blocks", type=int, default=25, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")
opt = parser.parse_args()
print(opt)
# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("checkpoints/%s" % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()  # 判别器的打分与实际的LOSS
criterion_MSE = torch.nn.MSELoss()  # 生成器生成的fake_A与real_A的LOSS
criterion_cycle = torch.nn.L1Loss()  #
criterion_identity = torch.nn.L1Loss()

cuda = torch.cuda.is_available()

input_shape = (opt.channels, opt.img_size, opt.img_size)

# Initialize generator and discriminator
G_BA = GeneratorResNet(input_shape, opt.n_residual_blocks)
D_A = Discriminator(input_shape)

if cuda:
    torch.cuda.set_device(2)  # 设置默认使用显卡
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_BA.load_state_dict(torch.load("checkpoints/%s/G_BA_%d.pth" % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load("checkpoints/%s/D_A_.pth" % (opt.dataset_name)))
else:
    # Initialize weights
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_BA.parameters()), lr=opt.lr_G, betas=(opt.b1, opt.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr_D, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations

train, test = split_TrainTest("./data/New-Total-Train/for_train/data.csv", "-14", "37")
config = {"use_gama": [0.4, 0.6],
          "use_flip": True,
          "use_random_add_intensity": True}
trainset = TrainDataset(train, "./data/New-Total-Train/for_train", opt.img_size, opt.channels, config=config, flag="not abs")
trainloader = DataLoader(
    trainset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True
)
testset = TrainDataset(test, "./data/New-Total-Train/for_train", opt.img_size, opt.channels, config=config, flag="not abs")
testloader = DataLoader(
    testset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True
)

yapianset = yapianDataset("./data/YaPian/raw2", opt.img_size, 3, 53)
yapianloader = DataLoader(
    yapianset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True
)


def sample_images(batches_done, flag):
    """Saves a generated sample from the test set"""
    if flag == "train":
        imgs = next(iter(trainloader))
    elif flag == "yapian":
        imgs = next(iter(yapianloader))
    elif flag == "test":
        imgs = next(iter(testloader))

    G_BA.eval()
    real_A = (imgs["A"].type(Tensor) + 1) / 2
    real_B = (imgs["B"].type(Tensor) + 1) / 2
    fake_A = (G_BA(real_B) + 1) / 2


    # Arange images along x-axis
    real_A = make_grid(real_A, nrow=5, normalize=False)
    real_B = make_grid(real_B, nrow=5, normalize=False)

    fake_A = make_grid(fake_A, nrow=5, normalize=False)

    image_grid = torch.cat((real_B, fake_A, real_A), 1)
    save_image(image_grid, "images/%s/%s.png" % (opt.dataset_name, batches_done), normalize=False)


# ----------
#  Training
# ----------

prev_time = time.time()
MSE_loss_min = 100
Loss = [[], []]
for epoch in range(opt.epoch, opt.n_epochs):
    flag = 0
    Mean_Gan_loss = 0
    temp_Loss = [[], []]

    for i, batch in enumerate(trainloader):
        flag += 1
        # Set model input
        real_A = batch["A"].type(Tensor)
        real_B = batch["B"].type(Tensor)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------

        G_BA.train()
        D_A.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_B), real_A)

        loss_identity = loss_id_A

        # GAN loss

        fake_A = G_BA(real_B)
        loss_GAN = criterion_GAN(D_A(fake_A), valid)
        loss_MSE = criterion_MSE(fake_A, real_A)
        #Mean_Gan_loss += loss_GAN / opt.batch_size    # 计算平均Gan损失

        # Total loss
        loss_G = 5 * loss_GAN + 30 * loss_identity + 100 * loss_MSE

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D_A.step()


        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(trainloader) + i
        batches_left = opt.n_epochs * len(trainloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if i % 20 == 0:
            print(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, GAN: %f, identity: %f, MSE:%f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(trainloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_GAN.item(),
                    loss_identity.item(),
                    loss_MSE.item(),
                    time_left,
                )
                            )
            temp_Loss[0].append(loss_MSE.item())

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, "train")

    # Update learning rates
    Mean_Gan_loss = Mean_Gan_loss / flag
    if MSE_loss_min > Mean_Gan_loss:
        MSE_loss_min = Mean_Gan_loss
    if epoch % 10 == 9:
        torch.save(G_BA.state_dict(), "checkpoints/%s/G_BA_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), "checkpoints/%s/D_A_.pth" % (opt.dataset_name))

    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    ##### TestLoader #####
    Test_G, Test_MSE, Test_Gan, LEN = 0, 0, 0, 0
    torch.cuda.empty_cache()
    for i, batch in enumerate(testloader):
        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False)

        G_BA.eval()
        D_A.eval()
        with torch.no_grad():
            # Identity loss
            fake_A = G_BA(real_B)
            loss_id_A = criterion_identity(fake_A, real_A)
            loss_identity = loss_id_A

            # GAN loss

            loss_GAN = criterion_GAN(D_A(fake_A), valid)
            loss_MSE = criterion_MSE(fake_A, real_A)

            # Total loss
            loss_G = 3 * loss_GAN + 10 * loss_identity + 20 * loss_MSE
            temp_Loss[1].append(loss_MSE.item())
        if i == 0:
            sample_images(epoch, "test")

    Loss[0].append(np.mean(np.array(temp_Loss[0])))
    Loss[1].append(np.mean(np.array(temp_Loss[1])))
    print(
        "\rTEST----[Epoch %d/%d] [MSE:%f]"
        % (
            epoch,
            opt.n_epochs,
            Loss[1][-1],
        )
    )
    if epoch % 5 == 0 and epoch > 2:
        from libs.visulize_results import *
        plot_result(Loss, opt, "MSE")

    ##### YaPianLoader #####
    torch.cuda.empty_cache()
    for i, batch in enumerate(yapianloader):
        G_BA.eval()
        D_A.eval()
        sample_images("yapian/%d_%d" % (epoch, i), "yapian")
