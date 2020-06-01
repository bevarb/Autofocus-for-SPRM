import argparse
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch

from models.cycle_models import *
import torch.nn as nn
from data.dataset import ImageDataset

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")

parser.add_argument("--dataset_name", type=str, default="DisFocus2Focus", help="name of the dataset")

parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--img_channels", type=int, default=3, help="number of image channels")

parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space 潜在空间的维数")

parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")

parser.add_argument("--sample_interval", type=int, default=1000, help="number of image channels")

opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
print("cuda_state:", cuda)




# !!! Minimizes MSE instead of BCE
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
# generator = Generator(opt)
# discriminator = Discriminator(opt)
input_shape = (opt.img_channels, opt.img_height, opt.img_width)
generator = GeneratorResNet(input_shape, opt.n_residual_blocks)

discriminator = Discriminator(input_shape)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()


# Configure data loader
transforms_ = [
#    transforms.Resize(opt.img_size),
#    transforms.RandomCrop((opt.img_height, opt.img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
dataset = ImageDataset("./Data/A", "./Data/B", transforms_=transforms_)
dataloader = DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# # Learning rate update schedulers
# lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )
# lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
#     optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
# )



# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))

        # print(imgs.shape)
        # Adversarial ground truths
        valid = Variable(Tensor(real_A.shape[0], 1).fill_(1.0), requires_grad=False)  # 虚构的真实图片
        fake = Variable(Tensor(real_A.shape[0], 1).fill_(0.0), requires_grad=False)  # 虚构的虚假图片

        # Configure input
        real_imgs = Variable(real_A.type(Tensor))  # 实际读取的图片，并转为Tensor格式

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()  #  将梯度清0

        # # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))  # 生成噪声图像作为generator的输入

        # Generate a batch of images
        gen_imgs = generator(real_B)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(generator.state_dict(), "checkpoints/%s/generator%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "checkpoints/%s/discriminator%d.pth" % (opt.dataset_name, epoch))
