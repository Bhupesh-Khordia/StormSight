from restormer.model import RestormerDerainer
import sys
import os

def main():
    if len(sys.argv) != 2:
        print("Usage: python derain_single_image.py /path/to/input_image.jpg")
        return

    input_path = sys.argv[1]

    if not os.path.exists(input_path):
        print(f"[ERROR] Image not found at {input_path}")
        return

    # Set output path
    output_path = "output_derained.jpg"

    # Initialize model
    derainer = RestormerDerainer()

    # Run deraining
    derainer.derain_image(input_path, output_path)
    print(f"[INFO] Derained image saved to: {output_path}")

if __name__ == "__main__":
    main()

