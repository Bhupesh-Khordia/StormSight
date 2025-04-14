import os
from glob import glob
from PIL import Image
from detection.detect_text import run_craft_text_detection
from detection.crop_img import crop_text_regions_from_images
from recognition.read_text import recognize_text_from_image

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
        # print(img_path)
        run_craft_text_detection( input_dir='../data/input', model_path='../models/detection/craft/craft_mlt_25k.pth', use_cuda=False)
        crop_text_regions_from_images()

cropped_dir = "../data/cropped"

output_dir = "../data/output"
os.makedirs(output_dir, exist_ok=True)

for img_file in os.listdir(cropped_dir):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(cropped_dir, img_file)
        text, conf = recognize_text_from_image(img_path)
        output_file = os.path.join(output_dir, "results.txt")
        with open(output_file, "a") as f:
            f.write(f"Image: {img_file}, Recognized text: {text}, Confidence: {conf}\n")