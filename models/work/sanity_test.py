import os
from utils import download_images_from_csv, preprocess_image

def sanity_test():
    # Get current working directory and construct dataset path
    current_dir = os.getcwd()
    dataset_path = os.path.join(current_dir, 'dataset', 'train.csv')
    image_dir = os.path.join(current_dir, 'images')
    
    # Debugging: Print current working directory and absolute dataset path
    print("Current working directory:", current_dir)
    print("Dataset path:", os.path.abspath(dataset_path))
    
    # Test downloading images
    download_images_from_csv(dataset_path, image_dir)
    
    # Check if images are downloaded and preprocessed
    if os.listdir(image_dir):
        print(f"Images downloaded to {image_dir}")
        
    # Preprocess the first sample image
    sample_image_path = os.path.join(image_dir, os.listdir(image_dir)[0])
    processed_image = preprocess_image(sample_image_path, (128, 128))
    print(f"Sample image preprocessed: {processed_image.size}")

if __name__ == "__main__":
    sanity_test()
