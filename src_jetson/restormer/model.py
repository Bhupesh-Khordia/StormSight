import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from runpy import run_path
from skimage import img_as_ubyte
import cv2
import os

class RestormerDerainer:
    def __init__(self, model_path="../models/deraining/restormer/deraining.pth", use_cuda=True):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.img_multiple_of = 8

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

        print("[INFO] Loading Restormer model...")
        self.model = self.load_model(model_path, parameters)
        print("[INFO] Model loaded and ready.")

    def load_model(self, weights_path, parameters):
        arch = run_path(os.path.join('restormer', 'basicsr', 'models', 'archs', 'restormer_arch.py'))
        model = arch['Restormer'](**parameters)
        checkpoint = torch.load(weights_path, map_location='cuda' if self.use_cuda else 'cpu')
        model.load_state_dict(checkpoint['params'])
        model.eval()
        if self.use_cuda:
            model = model.cuda()
        return model

    def derain_image(self, input_path, output_path):
        print(f"[INFO] Deraining: {input_path}")

        # Load and preprocess image
        img = cv2.cvtColor(cv2.imread(input_path), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))  # Optional resizing, adjust as needed

        input_tensor = torch.from_numpy(img).float().div(255.).permute(2, 0, 1).unsqueeze(0)
        if self.use_cuda:
            input_tensor = input_tensor.cuda()

        # Pad image to be multiple of 8
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        H = ((h + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        W = ((w + self.img_multiple_of) // self.img_multiple_of) * self.img_multiple_of
        padh = H - h
        padw = W - w
        input_tensor = F.pad(input_tensor, (0, padw, 0, padh), 'reflect')

        with torch.no_grad():
            restored = self.model(input_tensor)
            restored = torch.clamp(restored, 0, 1)
            restored = restored[:, :, :h, :w]
            restored = restored.permute(0, 2, 3, 1).cpu().numpy()[0]

        restored_img = img_as_ubyte(restored)
        output_dir = os.path.dirname(output_path)
        if output_dir:  # only make directories if path includes one
            os.makedirs(output_dir, exist_ok=True)

        cv2.imwrite(output_path, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

        print(f"[INFO] Output saved to: {output_path}")

