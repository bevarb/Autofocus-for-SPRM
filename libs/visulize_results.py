import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import numpy as np
def plot_result(LOSS, opt, method):

    x = [i for i in range(len(LOSS))]
    _plot_(x, np.array(LOSS), method, opt)

    # Train_Loss = pd.DataFrame(Train_Loss, columns=["Epoch", "Batch", "G_Loss", "D_Loss", "Gan_Loss", "L1_Loss", "L2_Loss"])
    # Train_Loss.to_csv("./images/Results/New-ROI-Train(move3)_test_loss/Train_Loss_ROI.csv", index=False)
    # Test_Loss = pd.DataFrame(Test_Loss, columns=["Epoch", "Batch", "Loss", "Gan_Loss", "L1_Loss", "L2_Loss"])
    # Test_Loss.to_csv("./images/Results/New-ROI-Train(move3)_test_loss/Test_Loss_ROI.csv", index=False)

def _plot_(x, Loss, Flag, opt):
    plt.plot(x[1:], Loss[1:, 2], label="Gan Loss")
    plt.plot(x[1:], Loss[1:, 3], label="L1 Loss")
    plt.plot(x[1:], Loss[1:, 4], label="L2 Loss")
    # ax = plt.gca()
    # ax.yaxis.set_major_locator(MultipleLocator(0.1))
    plt.legend()
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title(Flag)
    os.makedirs("./images/Results/New-ROI-Train(move3)_test_loss", exist_ok=True)
    plt.savefig("./images/Results/New-ROI-Train(move3)_test_loss/%s.png" % Flag)
    plt.close()