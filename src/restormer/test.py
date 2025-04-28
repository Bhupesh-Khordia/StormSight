import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
from tqdm import tqdm
import argparse
import numpy as np
import os
from model import return_model

input_dir = '../../data/test'
out_dir = '../../data/testOut'
os.makedirs(out_dir, exist_ok=True)
extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
files = natsorted(glob(os.path.join(input_dir, '*')))

# Check if files exist
if len(files) == 0:
    raise ValueError(f"No images found in {input_dir}. Check the path and that files exist.")

img_multiple_of = 8

print(f"\n ==> Running Restormer on CPU ...\n ")

# Load model ONCE before loop
model = return_model()
model.eval()

# No need for gradients
torch.set_grad_enabled(False)

for filepath in tqdm(files):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

    input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0)  # No .cuda()

    # Pad input to multiple of 8
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    # Inference
    restored = model(input_)
    restored = torch.clamp(restored, 0, 1)

    # Remove padding
    restored = restored[:, :, :h, :w]

    # Save output
    restored = restored.permute(0, 2, 3, 1).cpu().numpy()
    restored = img_as_ubyte(restored[0])

    filename = os.path.split(filepath)[-1]
    cv2.imwrite(os.path.join(out_dir, filename), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
