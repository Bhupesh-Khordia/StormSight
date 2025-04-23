import os
from glob import glob
from PIL import Image
from detection.detect_text import run_craft_text_detection
from detection.crop_img import crop_text_regions_from_images
from recognition.read_text import recognize_text_from_image
from deraining.single_image_test import derain_image_from_pipeline
import psutil
import time

def monitor_cpu_usage():
    while True:
        cpu_usage = psutil.cpu_percent(interval=1)
        with open("../data/output/cpu_log.txt", "a") as log_file:
            log_file.write(f"{time.ctime()} - CPU Usage: {cpu_usage}%\n")

if __name__ == "__main__":
    import threading
    monitor_thread = threading.Thread(target=monitor_cpu_usage, daemon=True)
    monitor_thread.start()

    print("Running the model pipeline...")

    input_dir = "../data/input"
    output_dir = "../data/derained"

    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        input_image_path = os.path.join(input_dir, img_file)
        output_image_path = os.path.join(output_dir, f"derained_{img_file}")

        # Call the deraining function for each image
        derain_image_from_pipeline(input_image_path, output_image_path)

    run_craft_text_detection(input_dir='../data/derained', model_path='../models/detection/craft/craft_mlt_25k.pth', use_cuda=False)
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