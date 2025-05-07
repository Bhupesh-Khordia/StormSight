# import torch
# import torch.nn.functional as F
# import torchvision.transforms.functional as TF
# from runpy import run_path
# from skimage import img_as_ubyte
# from natsort import natsorted
# from glob import glob
# import cv2
# from tqdm import tqdm
# import argparse
# import numpy as np
# import os

# def get_weights_and_parameters(parameters):
#     weights = "../../models/deraining/restormer/deraining.pth"
#     return weights, parameters

# # Get model weights and parameters
# parameters = {
#     'inp_channels':3, 
#     'out_channels':3, 
#     'dim':48, 
#     'num_blocks':[4,6,6,8], 
#     'num_refinement_blocks':4, 
#     'heads':[1,2,4,8], 
#     'ffn_expansion_factor':2.66, 
#     'bias':False, 
#     'LayerNorm_type':'WithBias', 
#     'dual_pixel_task':False
# }
# weights, parameters = get_weights_and_parameters(parameters)

# load_arch = run_path(os.path.join('basicsr', 'models', 'archs', 'restormer_arch.py'))
# model = load_arch['Restormer'](**parameters)
# # model.cuda()  # <-- COMMENTED OUT
# # Instead, keep it on CPU

# checkpoint = torch.load(weights, map_location=torch.device('cpu'))  # load to CPU
# model.load_state_dict(checkpoint['params'])
# model.eval()

# input_dir = '../data/input'
# out_dir = '../data/derained'
# os.makedirs(out_dir, exist_ok=True)
# extensions = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp', 'BMP']
# files = natsorted(glob(os.path.join(input_dir, '*')))

# # Check if files exist
# if len(files) == 0:
#     raise ValueError(f"No images found in {input_dir}. Check the path and that files exist.")

# img_multiple_of = 8

# print(f"\n ==> Running Restormer on CPU ...\n ")


# # No need for gradients
# torch.set_grad_enabled(False)

# for filepath in tqdm(files):
#     img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

#     input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0)  # No .cuda()

#     # Pad input to multiple of 8
#     h, w = input_.shape[2], input_.shape[3]
#     H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
#     padh = H - h if h % img_multiple_of != 0 else 0
#     padw = W - w if w % img_multiple_of != 0 else 0
#     input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

#     # Inference
#     restored = model(input_)
#     restored = torch.clamp(restored, 0, 1)

#     # Remove padding
#     restored = restored[:, :, :h, :w]

#     # Save output
#     restored = restored.permute(0, 2, 3, 1).cpu().numpy()
#     restored = img_as_ubyte(restored[0])

#     filename = os.path.split(filepath)[-1]
#     cv2.imwrite(os.path.join(out_dir, filename), cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))


import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
from natsort import natsorted
from glob import glob
import cv2
import os

# ----------------- Load Restormer model -----------------

use_cuda = True  # Set to True if you want to use GPU

if use_cuda:
    print("Warming up CUDA...")
    dummy = torch.randn(1).cuda()
    print("CUDA initialized successfully.")

if use_cuda:
    print("CUDA device:", torch.cuda.get_device_name(0))
    print("CUDA capability:", torch.cuda.get_device_capability(0))
    print("Memory total:", torch.cuda.get_device_properties(0).total_memory / (1024 ** 2), "MB")

def get_weights_and_parameters(parameters):
    weights = "../models/deraining/restormer/deraining.pth"
    return weights, parameters

parameters = {
    'inp_channels': 3, 
    'out_channels': 3, 
    'dim': 48, 
    'num_blocks': [4, 6, 6, 8], 
    'num_refinement_blocks': 4, 
    'heads': [1, 2, 4, 8], 
    'ffn_expansion_factor': 2.66, 
    'bias': False, 
    'LayerNorm_type': 'WithBias', 
    'dual_pixel_task': False,
}

weights, parameters = get_weights_and_parameters(parameters)

load_arch = run_path(os.path.join('restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
model = load_arch['Restormer'](**parameters)
if use_cuda:
    model.cuda()  # Move model to GPU

checkpoint = torch.load(weights)
model.load_state_dict(checkpoint['params'])
model.eval()

# ----------------- Define deraining function -----------------

def derain_single_image(input_path, output_path):
    img_multiple_of = 8

    print(f"\n ==> Running Restormer on {input_path} ...\n ")

    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    if use_cuda:
        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0).cuda()
    else:
        input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0)

    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - h if h % img_multiple_of != 0 else 0
    padw = W - w if w % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    if use_cuda:
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)
    else:
        with torch.no_grad():
            restored = model(input_)
            restored = torch.clamp(restored, 0, 1)

    restored = restored[:, :, :h, :w]

    if use_cuda:
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
    else:
        restored = restored.permute(0, 2, 3, 1).cpu().numpy()
    
    restored = img_as_ubyte(restored[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))
