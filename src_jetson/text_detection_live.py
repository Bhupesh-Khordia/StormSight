import os
import time
import torch
import cv2
import nanocamera as nano
from PIL import Image
from collections import OrderedDict
import numpy as np

import detection.craft_utils as craft_utils
import detection.imgproc as imgproc
from restormer.model import RestormerDerainer
from detection.craft import CRAFT
from recognition.strhub.data.module import SceneTextDataModule

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

print("[INFO] Loading CRAFT model...")
craft_model_path = '../models/detection/craft/craft_mlt_25k.pth'
craft_model = CRAFT()
craft_model.load_state_dict(copyStateDict(torch.load(craft_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')))
craft_model.eval()
if torch.cuda.is_available():
    craft_model = craft_model.cuda()
print("[INFO] CRAFT model loaded.")

print("[INFO] Loading Parseq model...")
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)
if torch.cuda.is_available():
    parseq = parseq.cuda()
print("[INFO] Parseq model loaded.")

def test_net_live(net, image, text_threshold=0.7, link_threshold=0.4, low_text=0.4,
                  canvas_size=720, mag_ratio=1.5, poly=False):
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(
        image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)

    if torch.cuda.is_available():
        x = x.cuda()

    with torch.no_grad():
        y, _ = net(x)

    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)

    return boxes

def recognize_text_from_image_array(image_array):
    try:
        img = Image.fromarray(image_array).convert('RGB')
        img = img_transform(img).unsqueeze(0)

        if torch.cuda.is_available():
            img = img.cuda()

        with torch.no_grad():
            logits = parseq(img)
            pred = logits.softmax(-1)
            label, confidence = parseq.tokenizer.decode(pred)

        return label[0], confidence[0]
    except Exception as e:
        print(f"[ERROR] Recognition failed: {e}")
        return None, None

def main():
    output_file = "live_results.txt"
    output_dir = "output"
    crops_dir = os.path.join(output_dir, "crops")
    frames_dir = os.path.join(output_dir, "frames")
    input_dir = os.path.join(output_dir, "input")
    derain_dir = os.path.join(output_dir, "derained")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(derain_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)

    csv_file = os.path.join(output_dir, "results.csv")
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            f.write("timestamp,region,text,confidence,crop_path,frame_path\n")

    print("[INFO] Initializing NanoCamera...")
    camera = nano.Camera(camera_type=1, device_id=0, width=640, height=480, fps=30)

    if not camera.isReady():
        print("[ERROR] Cannot open NanoCamera.")
        return

    derainer = RestormerDerainer()
    print("[INFO] Starting live text detection. Press 'q' to quit.")

    frame_count = 0
    skip_frames = 50
    last_boxes_and_texts = []

    while True:
        start_time = time.time()
        frame = camera.read()
        if frame is None:
            print("[WARNING] Failed to grab frame.")
            continue

        frame_count += 1
        timestamp = int(time.time() * 1000)

        if frame_count % skip_frames == 0:
            input_filename = f"{timestamp}.jpg"
            input_path = os.path.join(input_dir, input_filename)
            cv2.imwrite(input_path, frame)

            derain_filename = f"{timestamp}.jpg"
            derain_path = os.path.join(derain_dir, derain_filename)
            derainer.derain_image(input_path, derain_path)

            derained_frame = cv2.imread(derain_path)
            derained_frame = cv2.imread(input_path)
            if derained_frame is None:
                print("[ERROR] Failed to load derained image.")
                continue

            h_orig, w_orig = frame.shape[:2]
            h_der, w_der = derained_frame.shape[:2]

            scale_x = w_orig / w_der
            scale_y = h_orig / h_der

            boxes = test_net_live(craft_model, derained_frame[:, :, ::-1], canvas_size=144)

            print(f"[DEBUG] Detected {len(boxes)} boxes")

            new_boxes_and_texts = []

            for idx, box in enumerate(boxes):
                # Box is in 256x256 space (derained_frame)
                box = np.array(box)
                x_min = int(min(box[:, 0]))
                y_min = int(min(box[:, 1]))
                x_max = int(max(box[:, 0]))
                y_max = int(max(box[:, 1]))

                cropped_img = derained_frame[y_min:y_max, x_min:x_max]
                if cropped_img.size == 0:
                    continue

                text, conf = recognize_text_from_image_array(cropped_img)
                if text:
                    if torch.is_tensor(conf):
                        conf_val = conf.item() if conf.numel() == 1 else conf.mean().item()
                    elif isinstance(conf, (list, tuple)):
                        conf_val = float(conf[0]) if len(conf) == 1 else sum(conf) / len(conf)
                    else:
                        conf_val = float(conf)

                    print(f"Detected: {text} (Conf: {conf_val:.2f})")

                    crop_filename = f"{timestamp}_region{idx}.jpg"
                    crop_path = os.path.join(crops_dir, crop_filename)
                    cv2.imwrite(crop_path, cropped_img)

                    frame_filename = f"{timestamp}.jpg"
                    frame_path = os.path.join(frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)

                    with open(csv_file, "a") as f:
                        f.write(f"{timestamp},{idx},{text},{conf_val},{crop_path},{frame_path}\n")
                    with open(output_file, "a") as f:
                        f.write(f"Time-{timestamp} Region-{idx}: Text: {text}, Confidence: {conf_val}\n")

                    # Scale box for display on original frame
                    scaled_box = np.array([[int(x * scale_x), int(y * scale_y)] for x, y in box])
                    new_boxes_and_texts.append((scaled_box, text, timestamp))
                else:
                    print("[INFO] No SCENE Text Detected.")

            if new_boxes_and_texts:
                last_boxes_and_texts.extend(new_boxes_and_texts)

        # Prune old boxes
        current_time = int(time.time() * 1000)
        box_display_duration = 2000

        last_boxes_and_texts = [
            (box, text, ts) for box, text, ts in last_boxes_and_texts
            if current_time - ts <= box_display_duration
        ]

        for box, text, ts in last_boxes_and_texts:
            try:
                box = np.array(box)
                if box.shape == (4, 2):
                    pts = box.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
                    x_min = int(np.min(box[:, 0]))
                    y_min = int(np.min(box[:, 1]))
                    cv2.putText(frame, text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    print(f"[WARNING] Unexpected box shape: {box.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to draw box: {e}")

        # Show FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Live Text Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quitting...")
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
