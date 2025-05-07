import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import deraining.utils as utils
from deraining.model import MultiscaleNet as mynet
from deraining.get_parameter_number import get_parameter_number
from deraining.layers import *
from skimage import img_as_ubyte

def derain_image_from_pipeline(input_image_path, output_image_path, weights_path='../models/deraining/nerd_rain/model_large_SPA.pth', win_size=256):
    """
    Function to load the model, preprocess an image, run the deraining model, and save the output.

    Args:
    - input_image_path (str): Path to the input image to be derained.
    - output_image_path (str): Path to save the derained output image.
    - weights_path (str): Path to the pretrained model weights.
    - win_size (int): Size of the sliding window for image partitioning.

    Returns:
    - None: Saves the derained image to the output path.
    """
    # Load Model
    print('Loading model...')
    model_restoration = mynet()
    get_parameter_number(model_restoration)
    utils.load_checkpoint(model_restoration, weights_path)
    print("===> Testing using weights: ", weights_path)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_restoration.to(device)
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    # Preprocess Image
    def preprocess_image(image_path, win_size=256):
            img = cv2.imread(image_path)
            if img is None:
                print("Error: Image not found.")
                return
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
            img = cv2.resize(img, (256, 256))  # Resize image to 256x256
            img_tensor = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)
            return img_tensor, img.shape

    # Inference and Save Image
    def process_image(input_path, output_path, win_size=256):
        input_image, (Hx, Wx, _) = preprocess_image(input_path, win_size)
        input_re, batch_list = window_partitionx(input_image, win_size)

        with torch.no_grad():
            restored = model_restoration(input_re)
            restored = window_reversex(restored[0], win_size, Hx, Wx, batch_list)
            restored = torch.clamp(restored, 0, 1)
            restored = restored.squeeze().permute(1, 2, 0).cpu().numpy()
            restored_img = img_as_ubyte(restored)

        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))
        print(f"Derained image saved to: {output_path}")

    # Run the deraining process
    process_image(input_image_path, output_image_path, win_size)

