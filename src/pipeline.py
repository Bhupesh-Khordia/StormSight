import os
from glob import glob
from PIL import Image
from detection.detect_text import run_craft_text_detection
from detection.crop_img import crop_text_regions_from_images

input_dir = "../data/input"
# image_paths = glob(os.path.join(input_dir, "*.jpg"))  # Adjust extension if needed

# print(os.path.join(input_dir, "*.jpg"))

# for image_path in image_paths:
#     img = Image.open(image_path)
#     boxes = detect_text_regions(img, model_path="models/detection/craft/craft_mlt_25k.pth", use_cuda=False)

# print(os.listdir(input_dir))

for img_file in os.listdir(input_dir):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_dir, img_file)
        print(img_path)
        run_craft_text_detection( input_dir='../data/input', model_path='../models/detection/craft/craft_mlt_25k.pth', use_cuda=False)
        crop_text_regions_from_images()