# Image-Entity-Recognition

This project focuses on creating a machine learning model that extracts entity values from images. This capability is crucial in fields like healthcare, e-commerce, and content moderation, where precise product information is vital.

## Features
- Extracts entity values such as text, labels, and other attributes from images.
- Highly applicable in domains requiring accurate data extraction, such as healthcare (extracting patient info from scans), e-commerce (product data recognition), and content moderation.

## Major Libraries Used
- **Pillow**: For image manipulation and processing.
- **EasyOCR**: To perform optical character recognition (OCR) and extract text from images.
- **Pandas**: For data manipulation and managing datasets.
- **PyTorch**: As the deep learning framework for training and inference of the model.

## Dataset
- **Training Data**: 200,000 images from `train.csv`. Images are labeled with the entities to be recognized.
- **Testing Data**: 130,000 images from `test.csv`.
- The training and validation split is 60% for training and 40% for validation.

## Model Overview
- The model is built using PyTorch and is trained to recognize and extract entities from a wide variety of images.
- **Training**: The model was trained on 200,000 images for entity recognition.
- **Validation**: 40% of the training data was used for validation to fine-tune the model's performance.
- **Testing**: The model was tested on 130,000 images from the test set to evaluate its generalization ability.
