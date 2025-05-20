import os
import cv2
import numpy as np

def get_y_center(box):  # box is [x1, y1, x2, y2, ..., x4, y4]
    return np.mean([box[i+1] for i in range(0, 8, 2)])

def get_x_center(box):
    return np.mean([box[i] for i in range(0, 8, 2)])

def group_and_sort_boxes(boxes, line_thresh=15):
    boxes = sorted(boxes, key=get_y_center)
    lines = []
    current_line = [boxes[0]]

    for box in boxes[1:]:
        if abs(get_y_center(box) - get_y_center(current_line[0])) < line_thresh:
            current_line.append(box)
        else:
            lines.append(sorted(current_line, key=get_x_center))
            current_line = [box]
    lines.append(sorted(current_line, key=get_x_center))
    return [box for line in lines for box in line]

def crop_text_regions_from_images(image_folder="../../data/derained", result_folder="../../data/detected", output_folder="../../data/cropped"):
    os.makedirs(output_folder, exist_ok=True)

    for img_file in os.listdir(image_folder):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(image_folder, img_file)
        result_path = os.path.join(result_folder, f"res_{os.path.splitext(img_file)[0]}.txt")

        image = cv2.imread(img_path)
        if image is None:
            print(f"[ERROR] Failed to load image: {img_file}")
            continue

        height, width = image.shape[:2]

        if not os.path.exists(result_path):
            print(f"[WARNING] No result file for: {img_file}")
            continue

        boxes = []
        with open(result_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                try:
                    coords = list(map(int, parts[:8]))
                    boxes.append(coords)
                except ValueError:
                    print(f"[WARNING] Invalid coordinate values in {img_file}")
                    continue

        if not boxes:
            print(f"[WARNING] No valid boxes in {img_file}")
            continue

        # Sort boxes in reading order
        sorted_boxes = group_and_sort_boxes(boxes)

        for idx, coords in enumerate(sorted_boxes):
            points = np.array(coords, dtype=np.int32).reshape((4, 2))
            x, y, w, h = cv2.boundingRect(points)

            # Clip to image boundary
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                print(f"[WARNING] Invalid bounding box in {img_file}, word {idx+1}")
                continue

            cropped = image[y:y+h, x:x+w]

            crop_filename = f"{os.path.splitext(img_file)[0]}_word{idx+1}.jpg"
            crop_path = os.path.join(output_folder, crop_filename)
            cv2.imwrite(crop_path, cropped)

        print(f"[INFO] Processed {img_file}")