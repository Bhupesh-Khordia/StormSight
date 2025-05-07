import os
from detection.detect_text import run_craft_text_detection
from detection.crop_img import crop_text_regions_from_images
from recognition.read_text import recognize_text_from_image
import psutil
import time

if __name__ == "__main__":
    print("Running the model pipeline...")

    input_dir = "../data/input"

    # Text detection
    run_craft_text_detection(
        input_dir=input_dir,
        model_path='../models/detection/craft/craft_mlt_25k.pth',
        use_cuda=False
    )

    # Crop text regions
    crop_text_regions_from_images()

    cropped_dir = "../data/cropped"
    output_dir = "../data/output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "results.txt")

    for img_file in os.listdir(cropped_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(cropped_dir, img_file)
            text, conf = recognize_text_from_image(img_path)
            with open(output_file, "a") as f:
                f.write(f"Image: {img_file}, Recognized text: {text}, Confidence: {conf}\n")
        with open(output_file, "a") as f:
            f.write("\n")
