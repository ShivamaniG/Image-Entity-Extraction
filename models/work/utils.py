# utils.py

import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
import time
from requests.exceptions import Timeout

def download_image(url, save_path, retries=3, delay=5, timeout=10):
    if os.path.exists(save_path):
        print(f"Image already exists: {save_path}")
        return False  # Return False if the image is a duplicate

    for i in range(retries):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()  # Check if the request was successful
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image downloaded: {save_path}")
            return True  # Return True if the download was successful
        except Timeout:
            print(f"Timeout occurred for {url}, retrying ({i+1}/{retries})...")
            time.sleep(delay)
        except requests.exceptions.RequestException as e:
            print(f"Error occurred while downloading {url}: {e}")
            break

    return False

def load_dataset(csv_path):
    return pd.read_csv(csv_path)

def preprocess_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    return img

def download_images_from_csv(csv_path, save_dir):
    data = load_dataset(csv_path)
    os.makedirs(save_dir, exist_ok=True)
    for i, row in data.iterrows():
        image_url = row['image_link']
        save_path = os.path.join(save_dir, f"{row['group_id']}.jpg")
        download_image(image_url, save_path)
