import os
import pandas as pd
import re
import csv
import concurrent.futures
import easyocr
from utils import download_image, preprocess_image, load_dataset
from constants import ENTITY_UNIT_MAP

# Updated extract entity function
def extract_entity_value(text, entity_name):
    """Extracts entity value based on entity name from the text."""
    patterns = {
        'item_weight': r'(\d+(\.\d+)?)\s?(kg|kilogram|kg|kilo|g|gram|g|pound|lbs|Ibs|ounce|oz|ton|metric ton|milligram|mg|microgram|µg|KG|KILOGRAM|LB|POUND|OUNCE|TON|MILLIGRAM|MICROGRAM)\b',
        'maximum_weight_recommendation': r'(\d+(\.\d+)?)\s?(kg|kilogram|kg|kilo|g|gram|g|pound|lbs|ibs|ounce|oz|ton|metric ton|milligram|mg|microgram|µg|KG|KILOGRAM|LB|POUND|OUNCE|TON|MILLIGRAM|MICROGRAM)\b',
        'item_volume': r'(\d+(\.\d+)?)\s?(cup|liter|litre|l|ml|millilitre|gallon|gal|imperial gallon|centilitre|cl|microlitre|µl|pint|pt|quart|qt|CUP|LITER|LITRE|ML|MILLILITRE|GALLON|CENTILITRE|MICROLITRE|PINT|QUART)\b',
        'width': r'(\d+(\.\d+)?)\s?(cm|centimetre|cm|centi|mm|millimetre|mm|inch|in|In|ft|foot|metre|meter|m|yard|yd|CM|CENTIMETRE|INCH|FT|FOOT|METRE|METER|YARD)\b',
        'height': r'(\d+(\.\d+)?)\s?(cm|centimetre|cm|centi|mm|millimetre|mm|inch|in|In|ft|foot|metre|meter|m|yard|yd|CM|CENTIMETRE|INCH|FT|FOOT|METRE|METER|YARD)\b',
        'depth': r'(\d+(\.\d+)?)\s?(cm|centimetre|cm|centi|mm|millimetre|mm|inch|in|In|ft|foot|metre|meter|m|yard|yd|CM|CENTIMETRE|INCH|FT|FOOT|METRE|METER|YARD)\b',
        'voltage': r'(\d+(\.\d+)?)\s?(kilovolt|kv|kilovolt|kV|millivolt|mv|millivolt|mV|volt|v|volt|V|KILOVOLT|KV|MILLIVOLT|MV|VOLT|V)\b',
        'wattage': r'(\d+(\.\d+)?)\s?(kilowatt|kw|kilowatt|kW|watt|w|watt|W|KILOWATT|KW|WATT|W)\b'
    }

    entity_patterns = patterns.get(entity_name, "")

    if entity_patterns:
        # Find all matches in the text
        matches = re.findall(entity_patterns, text, re.IGNORECASE)

        if entity_name in ['item_weight', 'maximum_weight_recommendation']:
            item_weight_pattern = patterns['item_weight']
            max_weight_pattern = patterns['maximum_weight_recommendation']
            
            item_weight_matches = re.findall(item_weight_pattern, text, re.IGNORECASE)
            max_weight_matches = re.findall(max_weight_pattern, text, re.IGNORECASE)
            
            if entity_name == 'item_weight':
                # Assume item weight is listed first
                if item_weight_matches:
                    return f"{item_weight_matches[0][0]} {item_weight_matches[0][2]}"
                if max_weight_matches:
                    return f"{max_weight_matches[0][0]} {max_weight_matches[0][2]}"
            
            if entity_name == 'maximum_weight_recommendation':
                # Assume maximum weight recommendation is listed after item weight
                if max_weight_matches:
                    return f"{max_weight_matches[0][0]} {max_weight_matches[0][2]}"
                if item_weight_matches:
                    return f"{item_weight_matches[0][0]} {item_weight_matches[0][2]}"

        elif entity_name in ['width', 'height', 'depth']:
            dimensions = {'height': None, 'width': None, 'depth': None}

            # Assign matches to height, width, and depth
            if len(matches) > 0:
                dimensions['height'] = f"{matches[0][0]} {matches[0][2]}"
            if len(matches) > 1:
                dimensions['width'] = f"{matches[1][0]} {matches[1][2]}"
            if len(matches) > 2:
                dimensions['depth'] = f"{matches[2][0]} {matches[2][2]}"

            if dimensions[entity_name] is not None:
                return dimensions[entity_name]
            else:
                # Return the single available value if only one dimension was found
                non_null_values = {key: value for key, value in dimensions.items() if value is not None}
                if non_null_values:
                    # Return the first available value if the specific entity is not found
                    return list(non_null_values.values())[0]

        elif entity_name in ['voltage', 'wattage']:
            # Handle voltage and wattage separately
            voltage_pattern = patterns['voltage']
            wattage_pattern = patterns['wattage']
            
            voltage_matches = re.findall(voltage_pattern, text, re.IGNORECASE)
            wattage_matches = re.findall(wattage_pattern, text, re.IGNORECASE)
            
            if entity_name == 'voltage':
                if voltage_matches:
                    return f"{voltage_matches[0][0]} {voltage_matches[0][2]}"
                if wattage_matches:
                    return f"{wattage_matches[0][0]} {wattage_matches[0][2]}"
            
            if entity_name == 'wattage':
                if wattage_matches:
                    return f"{wattage_matches[0][0]} {wattage_matches[0][2]}"
                if voltage_matches:
                    return f"{voltage_matches[0][0]} {voltage_matches[0][2]}"

        else:
            # Handle other entity names
            if matches:
                number = matches[0][0]  # Extract the number from the first match
                unit = matches[0][2]    # Extract the unit from the first match
                return f"{number} {unit}"

    return ""

def process_image(row, reader, image_dir):
    idx = row.name  # Get the index from the DataFrame row
    image_url = row['image_link']
    entity_name = row['entity_name']
    group_id = row['group_id']

    # Step 1: Define image path
    image_path = os.path.join(image_dir, f"{group_id}.jpg")

    # Check if image already exists
    if os.path.exists(image_path):
        print(f"Image already exists: {image_path}")
    else:
        # Step 2: Download image if it does not exist
        if not download_image(image_url, image_path):
            print(f"Skipping image {group_id} due to download failure.")
            return [idx, ""]

    # Step 3: Preprocess image
    img = preprocess_image(image_path, (128, 128))

    # Step 4: Use EasyOCR to extract text
    extracted_text = reader.readtext(image_path, detail=0)
    extracted_text = ' '.join(extracted_text)  # Convert list to a single string
    print(f"Extracted Text for {group_id}.jpg: {extracted_text}")

    print(f"Entity Name: {entity_name}")

    # Step 5: Extract entity value
    entity_value = extract_entity_value(extracted_text, entity_name)
    print(f"Entity Value for {group_id}.jpg: {entity_value}")

    # Return result
    return [idx, entity_value]

def run_extraction_test(csv_path, image_dir, output_csv):
    # Load test dataset
    test_data = load_dataset(csv_path)

    # Initialize EasyOCR reader with GPU if available
    reader = easyocr.Reader(['en'], gpu=True)  # Enable GPU for faster processing

    # Create output list to store predictions
    output = []

    # Use ThreadPoolExecutor to parallelize image processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit each row of the test_data for processing
        futures = [executor.submit(process_image, row, reader, image_dir) for _, row in test_data.iterrows()]
        
        # Gather results as futures complete
        for future in concurrent.futures.as_completed(futures):
            output.append(future.result())

    # Step 7: Save results to CSV
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'prediction'])
        writer.writerows(output)

    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    dataset_path = 'dataset/test.csv'  # Test CSV path
    image_dir = 'images_test/'  # Directory to save images
    output_csv = 'output_predictions.csv'  # Output CSV file

    os.makedirs(image_dir, exist_ok=True)
    run_extraction_test(dataset_path, image_dir, output_csv)
