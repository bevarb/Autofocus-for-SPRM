import os
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from PIL import Image
from skimage.measure import compare_mse, compare_psnr, compare_ssim
import numpy as np
from utils import *
class plot_test():
    def __init__(self, result, Flag):
        self.data = result
        self.Flag = Flag
        self.len = len(self.data)

    def evaluate_(self):
        mse, psnr, ssim = np.zeros([2, self.len]), np.zeros([2, self.len]), np.zeros([2, self.len])
        for i in range(self.len):
            real_A = self.data[i][0].numpy()[0]
            real_B = self.data[i][1].numpy()[0]
            fake_A = self.data[i][2].cpu().detach().numpy()[0]
            from sklearn.metrics import mean_squared_error

            mse_input = compare_mse(real_B, real_A)
            mse_output = compare_mse(fake_A, real_A)

            ssim_input = compare_ssim(real_B[0, :, :], real_A[0, :, :], multichannel=True)
            ssim_output = compare_ssim(fake_A[0, :, :], real_A[0, :, :], multichannel=True)

            mse[:, i] = [mse_input, mse_output]
            ssim[:, i] = [ssim_input, ssim_output]
        plt.plot([i * 0.2 for i in range(self.len)], mse[0, :], label="Real_Image")
        plt.plot([i * 0.2 for i in range(self.len)], mse[1, :], label="Gan_Image")
        plt.title("MSE")
        # plt.ylim(0, 1)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(0.4))
        plt.legend()
        plt.xlabel("Focus")
        plt.show()
        plt.plot([i * 0.2 for i in range(self.len)], ssim[0, :])
        plt.plot([i * 0.2 for i in range(self.len)], ssim[1, :])
        plt.title("SSIM")
        plt.show()
    def plot_fid(self, fid1_mean, fid1_std, fid2_mean, fid2_std, begain):
        plt.errorbar([i * 0.2 + begain * 0.2 for i in range(len(fid1_mean))], fid1_mean, fid1_std, label="Real_Image")
        plt.errorbar([i * 0.2 + begain * 0.2 for i in range(len(fid2_mean))], fid2_mean, fid2_std, label="Gan_Image")
        plt.title("FID")
        # plt.ylim(0, 1)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(MultipleLocator(0.4))
        plt.legend()
        plt.xlabel("Focus")
        plt.show()

    def show_results(self, step=1):
        real_A = Variable(self.data[0][0].type(torch.Tensor))
        real_B = Variable(self.data[0][1].type(torch.Tensor))
        fake_A = Variable(self.data[0][2].type(torch.Tensor))
        All = torch.cat((real_B, fake_A, real_A), 1)
        for i in range(self.len - 1):
            real_A = Variable(self.data[i+1][0].type(torch.Tensor))
            real_B = Variable(self.data[i+1][1].type(torch.Tensor))
            fake_A = Variable(self.data[i+1][2].type(torch.Tensor))
            error = torch.sub(real_A, fake_A)
            temp = torch.cat((real_B, fake_A, real_A, error), 1)
            All = torch.cat((All, temp), 0)
        save_image(All, "data/test/%s/result.png" % (self.Flag), normalize=False)

    # def calculate_fid(self, act1, act2):
    #     mu1, sigma1 = act1.mean(axis=0), act1.cov(act1, rowvar=False)
    #     mu2, sigma2 = act2.mean(axis=0), act2.cov(act2, rowvar=False)
    #     ssdiff = np.sum((mu1 - mu2)**2.0)
    #     covmean = np.sqrt(sigma1.dot(sigma2))






