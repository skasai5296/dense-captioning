import glob
import os

import torch

paths = sorted(
    glob.glob("/home/seito/hdd/dsets/activitynet_captions/features/resnet/*.pth")
)
n = len(paths)
for i, path in enumerate(paths):
    print("{}/{}".format(i + 1, n))
    torch.save(torch.load(path).t(), path)
