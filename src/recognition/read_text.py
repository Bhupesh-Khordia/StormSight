import torch
from PIL import Image
from recognition.strhub.data.module import SceneTextDataModule

# Load model only once (global scope so it's not reloaded every time the function is called)
parseq = torch.hub.load('baudm/parseq', 'parseq', pretrained=True).eval()
img_transform = SceneTextDataModule.get_transform(parseq.hparams.img_size)

def recognize_text_from_image(image_path):
    """
    Recognizes text from an image using the Parseq model.
    
    Parameters:
        image_path (str): Path to the image file.

    Returns:
        tuple: (decoded_text, confidence)
    """
    try:
        # Load and preprocess image
        img = Image.open(image_path).convert('RGB')
        img = img_transform(img).unsqueeze(0)

        # Inference
        with torch.no_grad():
            logits = parseq(img)
            pred = logits.softmax(-1)
            label, confidence = parseq.tokenizer.decode(pred)

        return label[0], confidence[0]
    
    except Exception as e:
        print(f"[ERROR] Failed to process image {image_path}: {e}")
        return None, None
