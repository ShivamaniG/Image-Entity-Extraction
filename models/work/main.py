import easyocr
import os
import re
from utils import download_images_from_csv, preprocess_image
from constants import ENTITY_UNIT_MAP

def extract_entity_value(text):
    """Extracts entity value based on entity name from the text."""
    patterns = {
        'item_weight': r'\b\d+(\.\d+)?\s?(kg|g|pound|lbs|ounce)\b',
        'item_volume': r'\b\d+(\.\d+)?\s?(cup|liter|ml|gallon)\b',
        'width': r'\b\d+(\.\d+)?\s?(cm|mm|inch)\b',
        'height': r'\b\d+(\.\d+)?\s?(cm|mm|inch)\b',
        'depth': r'\b\d+(\.\d+)?\s?(cm|mm|inch)\b',
    }
    
    for entity_name, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return entity_name, match.group()
    
    return "Unknown Entity", "Entity not found"

def run_extraction_pipeline(csv_path, image_dir):
    # Step 1: Download images
    # download_images_from_csv(csv_path, image_dir)
    
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)  # 'en' is for English, you can add other language codes

    # Step 2: Iterate over each image and extract text using OCR
    for image_file in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_file)
        img = preprocess_image(image_path, (128, 128))

        # Step 3: Use EasyOCR to extract text
        extracted_text = reader.readtext(image_path, detail=0)
        extracted_text = ' '.join(extracted_text)  # Convert list to a single string
        print(f"Extracted Text for {image_file}: {extracted_text}")
        
        # Step 4: Extract entity values
        entity_name, entity_value = extract_entity_value(extracted_text)
        print(f"Entity Name: {entity_name}")
        print(f"Entity Value: {entity_value}")

if __name__ == "__main__":
    dataset_path = 'dataset/train.csv'
    image_dir = 'images/'

    run_extraction_pipeline(dataset_path, image_dir)
