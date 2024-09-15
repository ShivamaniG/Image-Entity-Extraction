import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import os
from model import EntityExtractor

os.chdir(os.path.dirname(os.path.abspath(__file__)))

train_csv_path = 'train.csv'
test_csv_path = 'test.csv'
image_dir = 'images'
model_path = 'entity_extractor.pth'

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.df['image_link'].iloc[idx])
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            print(f"Error: Image not found - {image_path}")
            return torch.zeros(3, 224, 224), torch.tensor(-1)  # Return a dummy image and label

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(self.df['label'].iloc[idx])  # Assuming 'label' is numeric

        return image, label

def load_data(csv_path, image_dir, transform):
    df = pd.read_csv(csv_path)
    dataset = MyDataset(df, image_dir, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True if csv_path == train_csv_path else False)

def train(model, train_loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            if labels is None:  # Skip batch if images are missing
                continue
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            if labels is None:  # Skip batch if images are missing
                continue
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    num_classes = 10  
    image_size = (224, 224)
    num_epochs = 10
    learning_rate = 0.001

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load data
    train_loader = load_data(train_csv_path, image_dir, transform)
    test_loader = load_data(test_csv_path, image_dir, transform)

    # Create model
    model = EntityExtractor(num_classes=num_classes)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train(model, train_loader, optimizer, criterion, num_epochs)

    # Save the trained model
    torch.save(model.state_dict(), model_path)

    # Test the model
    test(model, test_loader)
