'''
Open the file:

<your_env_path>\Lib\site-packages\imgaug\augmenters\meta.py

Find line 3368:

augmenter_active = np.zeros((nb_rows, len(self)), dtype=np.bool)

Change it to:

augmenter_active = np.zeros((nb_rows, len(self)), dtype=bool)

'''


import os
import cv2
import shutil
from tqdm import tqdm
import imgaug.augmenters as iaa

# === Define your input/output paths ===
sets = [
    {
        "name": "train",
        "img_dir": "ICDAR2013/Challenge2_Training_Task12_Images",
        "gt_dir": "ICDAR2013/Challenge2_Training_Task1_GT",
        "output_dir": "rainy_icdar2013/train"
    },
    {
        "name": "test",
        "img_dir": "ICDAR2013/Challenge2_Test_Task12_Images",
        "gt_dir": "ICDAR2013/Challenge2_Test_Task1_GT (1)",
        "output_dir": "rainy_icdar2013/test"
    }
]

# === Rain Augmenter ===
rain = iaa.Sequential([
    iaa.Rain(speed=(0.1, 0.15), drop_size=(0.2, 0.3)),
    iaa.MotionBlur(k=3)  # optional: slight blur for motion effect
])


def add_rain(image):
    return rain(image=image)

def process_set(set_config):
    os.makedirs(set_config["output_dir"], exist_ok=True)

    image_files = [f for f in os.listdir(set_config["img_dir"]) if f.lower().endswith(".jpg")]

    for img_file in tqdm(image_files, desc=f"Processing {set_config['name']} set"):
        img_path = os.path.join(set_config["img_dir"], img_file)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rainy_rgb = add_rain(img_rgb)
        rainy_bgr = cv2.cvtColor(rainy_rgb, cv2.COLOR_RGB2BGR)

        # Save rainy image
        out_img_path = os.path.join(set_config["output_dir"], img_file)
        cv2.imwrite(out_img_path, rainy_bgr)

        # Copy annotation
        gt_file = "gt_" + os.path.splitext(img_file)[0] + ".txt"
        src_gt_path = os.path.join(set_config["gt_dir"], gt_file)
        dst_gt_path = os.path.join(set_config["output_dir"], gt_file)
        if os.path.exists(src_gt_path):
            shutil.copy(src_gt_path, dst_gt_path)
        else:
            print(f"[Warning] Missing annotation for {img_file}")

# === Run for both train and test sets ===
for s in sets:
    process_set(s)

print("\n✅ Rainy versions saved in: rainy_icdar2013/train and rainy_icdar2013/test")
