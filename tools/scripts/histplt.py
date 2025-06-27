import os
import time

import torch
import matplotlib.pyplot as plt


os.makedirs("pth/hist", exist_ok=True)
def histplt(input, title, bins=1000,):
    min, max = torch.floor(input.min()).int(), torch.ceil(input.max()).int()
    x = torch.linspace(min, max, bins)
    hist_map = torch.histc(input, bins, min, max).cpu().numpy()
    cum_hist_map = [hist_map[:i].sum() for i in range(bins)]

    # plot 1:
    plt.subplot(1, 2, 1,)
    plt.bar(x, hist_map, width=0.2)

    plt.title("Histogram", fontsize=8)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.gca().ticklabel_format(useMathText=True)
    plt.grid()

    # plot 2:
    plt.subplot(1, 2, 2)
    plt.bar(x, cum_hist_map, width=0.2)

    plt.title("Cumulative histogram", fontsize=8)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.gca().ticklabel_format(useMathText=True)
    plt.grid()

    plt.suptitle(title)

    plt.savefig("pth\hist\{}.jpg".format(time.time()), dpi=300)
    plt.show()
    plt.clf()
    plt.close()
