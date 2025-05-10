import os
import cv2
import csv
import json
from google import genai
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ==== CONFIG ====
INPUT_DIR = "./gemini"
OUTPUT_DIR = "./output_images"
CSV_PATH = "./results.csv"
API_KEY = os.getenv("GEMINI_API_KEY")  # Replace with your actual API key

# ==== SETUP ====
os.makedirs(OUTPUT_DIR, exist_ok=True)
client = genai.Client(api_key=API_KEY)  # Configure the API key

# ==== HELPERS ====
def parse_response(response_text):
    """
    Extract JSON from Gemini response text and parse it.
    Handles cases where the response might contain additional text.
    """
    try:
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("JSON array not found in response.")
        json_str = response_text[start_idx:end_idx + 1]
        return json.loads(json_str)
    except Exception as e:
        print(f"Failed to parse JSON: {e}. Response text: {response_text}")
        return []

def draw_boxes(image_path, detections, output_path, original_size):
    """
    Draw bounding boxes on the image, adjusting coordinates to the original size.

    Args:
        image_path (str): Path to the image.
        detections (list): List of detection dictionaries.
        output_path (str): Path to save the output image.
        original_size (tuple): (width, height) of the original image.
    """
    image = cv2.imread(image_path)
    h, w = original_size  # Unpack original width and height.  Important
    img_h, img_w, _ = image.shape # shape of the image.
    
    for detection in detections:
        x1, y1, x2, y2 = detection["box_2d"]
        label = detection["label"]

        # Adjust coordinates to the original image size.
        x1 = int(x1 * 5.568) # scale with respect to original image.
        y1 = int(y1 * 4.872)
        x2 = int(x2 * 5.568)
        y2 = int(y2 * 4.872)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.imwrite(output_path, image)

# ==== MAIN ====
def main():
    with open(CSV_PATH, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Image", "Detected Text", "Box Coordinates"])

        for image_file in os.listdir(INPUT_DIR):
            if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            img_path = os.path.join(INPUT_DIR, image_file)
            out_img_path = os.path.join(OUTPUT_DIR, image_file)

            print(f"Processing: {image_file}")
            try:
                # Load the image to get its original size *before* sending it to Gemini
                original_image = cv2.imread(img_path)
                original_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
                
                uploaded_file = client.files.upload(file=img_path)
                prompt = (
                    "Read scene text in this image, and if some characters of a word are hidden and "
                    "you find that detected word has no meaning, correct it. "
                    "Return output in this JSON format only: "
                    "[{\"box_2d\": [x1, y1, x2, y2], \"label\": \"TEXT\"}, ...]"
                )

                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=[uploaded_file, prompt]
                )

                detections = parse_response(response.text)
                draw_boxes(img_path, detections, out_img_path, original_size)  # Pass original size

                for det in detections:
                    writer.writerow([image_file, det["label"], det["box_2d"]])

            except Exception as e:
                print(f"Error processing {image_file}: {e}")

if __name__ == "__main__":
    main()
