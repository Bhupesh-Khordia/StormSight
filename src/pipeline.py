import os
from detection.detect_text import run_craft_text_detection
from detection.crop_img import crop_text_regions_from_images
from recognition.read_text import recognize_text_from_image
from restormer.model import derain_single_image 
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StormSight Model Pipeline")
    parser.add_argument("--input", type=str, default="../data/input", help="Directory with input images")
    parser.add_argument("--output", type=str, default="../data/output", help="Directory to save output results")
    args = parser.parse_args()

    print("Running the model pipeline...")

    os.makedirs(args.derained_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_file in image_files:
        input_image_path = os.path.join(args.input_dir, img_file)
        output_image_path = os.path.join(args.derained_dir, f"derained_{img_file}")

        # Derain each image
        derain_single_image(input_image_path, output_image_path)

    # Text detection
    run_craft_text_detection(
        input_dir=args.derained_dir,
        model_path=args.craft_model,
        use_cuda=args.use_cuda
    )

    # Crop text regions
    crop_text_regions_from_images()

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, args.output_file)

    for img_file in os.listdir(args.cropped_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(args.cropped_dir, img_file)
            text, conf = recognize_text_from_image(img_path)
            with open(output_file, "a") as f:
                f.write(f"Image: {img_file}, Recognized text: {text}, Confidence: {conf}\n")
        with open(output_file, "a") as f:
            f.write("\n")
