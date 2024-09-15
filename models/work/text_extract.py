import os
import pandas as pd
import csv
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
from utils import download_image, load_dataset

def preprocess_image(image_path):
    """Preprocesses the image to enhance text clarity."""
    with Image.open(image_path) as img:
        # Convert to grayscale
        img = img.convert('L')
        # Enhance contrast
        img = ImageEnhance.Contrast(img).enhance(2.0)
        # Apply thresholding to make text stand out
        img = img.point(lambda p: p > 128 and 255)
        # Save the preprocessed image
        preprocessed_path = image_path.replace('.jpg', '_processed.jpg')
        img.save(preprocessed_path)
        return preprocessed_path

def extract_text_from_image(image_path, reader):
    """Extracts text from an image using EasyOCR."""
    # Use EasyOCR to extract text
    extracted_text = reader.readtext(image_path, detail=0)
    # Convert list to a single string
    return ' '.join(extracted_text)

def run_extraction_test(csv_path, image_dir, output_csv):
    # Load test dataset
    test_data = load_dataset(csv_path)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'], gpu=False)  # English OCR

    # Create output list to store predictions
    output = []

    for idx, row in test_data.iterrows():
        image_url = row['image_link']
        group_id = row['group_id']

        # Define image path
        image_path = os.path.join(image_dir, f"{group_id}.jpg")
        
        # Check if image already exists
        if not os.path.exists(image_path):
            # Download image if it does not exist
            if not download_image(image_url, image_path):
                print(f"Skipping image {group_id} due to download failure.")
                output.append([idx, ""])
                continue

        # Preprocess image
        preprocessed_image_path = preprocess_image(image_path)

        # Extract text from the preprocessed image
        extracted_text = extract_text_from_image(preprocessed_image_path, reader)
        print(f"Extracted Text for {group_id}.jpg: {extracted_text}")

        # Store extracted text in output list
        output.append([idx, extracted_text])

    # Save results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'text_extracted'])
        writer.writerows(output)

    print(f"Extracted texts saved to {output_csv}")

if __name__ == "__main__":
    dataset_path = 'dataset/test.csv'  # Test CSV path
    image_dir = 'images_test/'  # Directory to save images
    output_csv = 'output_text_extractions.csv'  # Output CSV file

    os.makedirs(image_dir, exist_ok=True)
    run_extraction_test(dataset_path, image_dir, output_csv)
