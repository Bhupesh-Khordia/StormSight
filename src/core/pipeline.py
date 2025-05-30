import os
import sys
sys.path.append("..")
from detection.detect_text import run_craft_text_detection
from detection.crop_img import crop_text_regions_from_images
from recognition.read_text import recognize_text_from_image
from restormer.model import derain_single_image 
from restormer.model import load_restormer_model
import argparse
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StormSight Model Pipeline")
    parser.add_argument("--input", type=str, default="../../data/inputs", help="Directory with input images")
    parser.add_argument("--output", type=str, default="../../data/output", help="Directory to save output results")
    args = parser.parse_args()

    print("Running the model pipeline...")

    os.makedirs("../../data/derained", exist_ok=True)

    image_files = [f for f in os.listdir(args.input) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    restormer_model = load_restormer_model()

    for img_file in image_files:
        input_image_path = os.path.join(args.input, img_file)
        output_image_path = os.path.join("../../data/derained", f"derained_{img_file}")

        # Derain each image
        derain_single_image(input_image_path, output_image_path, restormer_model)

    # Text detection
    run_craft_text_detection(
        input_dir="../../data/derained",
        model_path="../../models/detection/craft_mlt_25k.pth",
        use_cuda=False,
        result_folder="../../data/detected/",
    )

    # Crop text regions
    crop_text_regions_from_images(image_folder='../../data/derained', result_folder="../../data/detected", output_folder="../../data/cropped")

    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, "results.txt")

    def extract_line_index(filename):
        match = re.search(r'_word(\d+)', filename)
        return int(match.group(1)) if match else float('inf')

    cropped_images = sorted(
        [f for f in os.listdir("../../data/cropped") if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        key=extract_line_index
    )
    for img_file in cropped_images:
        img_path = os.path.join("../../data/cropped", img_file)
        text, conf = recognize_text_from_image(img_path)
        with open(output_file, "a") as f:
            f.write(f"Image: {img_file}, Recognized text: {text}, Confidence: {conf}\n")
    with open(output_file, "a") as f:
        f.write("\n")
