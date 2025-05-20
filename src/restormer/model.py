import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
import cv2
import os

# Global config
use_cuda = False  # Set to True if CUDA is available

def load_restormer_model():
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

    weights_path = "../../models/deraining/deraining.pth"
    model_arch = run_path(os.path.join('..', 'restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
    model = model_arch['Restormer'](**parameters)

    if use_cuda:
        model = model.cuda()

    checkpoint = torch.load(weights_path, map_location="cuda" if use_cuda else "cpu")
    model.load_state_dict(checkpoint['params'])
    model.eval()

    return model

def derain_single_image(input_path, output_path, model):
    img_multiple_of = 8

    print(f"\n ==> Running Restormer on {input_path} ...\n ")
    img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
    input_ = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0)
    if use_cuda:
        input_ = input_.cuda()

    h, w = input_.shape[2:]
    H, W = ((h + img_multiple_of) // img_multiple_of) * img_multiple_of, ((w + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh, padw = H - h, W - w
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    with torch.no_grad():
        restored = model(input_)
        restored = torch.clamp(restored, 0, 1)

    restored = restored[:, :, :h, :w].permute(0, 2, 3, 1).cpu().numpy()
    restored = img_as_ubyte(restored[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(restored, cv2.COLOR_RGB2BGR))