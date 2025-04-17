import os
import cv2
import numpy as np

def crop_text_regions_from_images(image_folder="../data/derained", result_folder="../data/detected", output_folder="../data/cropped"):
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(image_folder):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(image_folder, img_file)
            result_path = os.path.join(result_folder, f"res_{os.path.splitext(img_file)[0]}.txt")

            image = cv2.imread(img_path)
            if image is None:
                print(f"[ERROR] Failed to load image: {img_file}")
                continue

            if not os.path.exists(result_path):
                print(f"[WARNING] No result file for: {img_file}")
                continue

            with open(result_path, 'r') as f:
                lines = f.readlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue

                coords = list(map(int, parts[:8]))
                points = np.array(coords, dtype=np.int32).reshape((4, 2))
                x, y, w, h = cv2.boundingRect(points)
                cropped = image[y:y+h, x:x+w]

                crop_filename = f"{os.path.splitext(img_file)[0]}_line{idx+1}.jpg"
                crop_path = os.path.join(output_folder, crop_filename)
                cv2.imwrite(crop_path, cropped)

            print(f"[INFO] Processed {img_file}")

