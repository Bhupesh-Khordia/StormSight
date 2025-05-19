import os
import re
from difflib import SequenceMatcher
from glob import glob

# == Load ground truth ==
def load_ground_truth(gt_folder):
    gt_dict = {}
    for file in glob(os.path.join(gt_folder, 'gt_*.txt')):
        with open(file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            gt_entries = []
            for line in lines:
                parts = line.strip().split(',')
                if len(parts) >= 5:
                    text = parts[-1].strip().strip('"').lower()
                    gt_entries.append(text)
        fname = os.path.splitext(os.path.basename(file))[0]
        img_id = fname.replace('gt_', '')
        gt_dict[img_id] = gt_entries
    return gt_dict

# == Load predictions ==
def load_predictions(output_file):
    pred_dict = {}
    with open(output_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            match = re.match(r'Image: (.*?), Recognized text: (.*?), Confidence:.*', line)
            if match:
                image_name, recognized_text = match.groups()
                base_match = re.match(r'.*?img_(\d+)_line\d+', image_name)
                if base_match:
                    img_id = f'img_{base_match.group(1)}'
                    if img_id not in pred_dict:
                        pred_dict[img_id] = []
                    pred_dict[img_id].append(recognized_text.strip().lower())
    return pred_dict

# == Match one GT entry to best unmatched prediction ==
def best_match(gt, predictions, threshold=0.9):
    best_score = 0
    best_idx = -1
    for i, pred in enumerate(predictions):
        ratio = SequenceMatcher(None, gt, pred).ratio()
        if ratio > best_score:
            best_score = ratio
            best_idx = i
    if best_score >= threshold:
        return best_idx
    return -1

# == Evaluation ==
def evaluate(gt_dict, pred_dict, log_file='../data/output/evaluation_log.txt'):
    total = 0
    correct = 0
    logs = []

    logs.append("📊 Evaluation Results:\n------------------------")

    for img_id in sorted(gt_dict.keys()):
        gt_texts = gt_dict[img_id]
        pred_texts = pred_dict.get(img_id, []).copy()
        matched_preds = [False] * len(pred_texts)

        correct_img = 0
        mismatches = []
        for gt_text in gt_texts:
            total += 1
            match_idx = best_match(gt_text, pred_texts)
            if match_idx != -1 and not matched_preds[match_idx]:
                correct += 1
                correct_img += 1
                matched_preds[match_idx] = True
            else:
                mismatches.append(gt_text)

        logs.append(f"[{img_id}] ✅ {correct_img}/{len(gt_texts)} correct")
        if mismatches:
            logs.append(f"    ❌ Missed GT words: {', '.join(mismatches)}")

    accuracy_str = "\n🧮 Overall Accuracy: {}/{} = {:.2%}".format(correct, total, correct / total)
    logs.append(accuracy_str)

    # Print to console
    for line in logs:
        print(line)

    # Save to file
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(logs))

    print(f"\n📁 Log saved to: {log_file}")



gt_path = '../data/input'  # e.g. './Challenge2_Training_Task1_GT'
output_txt = '../data/output/results.txt'           # e.g. './output.txt'


gt_dict = load_ground_truth(gt_path)
pred_dict = load_predictions(output_txt)
evaluate(gt_dict, pred_dict)
