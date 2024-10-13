import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class ChessDataset(Dataset):
    def __init__(self, images_folder, labels_file, img_size=(128, 128), transform=None):
        self.images_folder = images_folder
        self.labels = pd.read_excel(labels_file)
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.labels.iloc[idx]['name']
        image_path = os.path.join(self.images_folder, image_name)
        
        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            label = self.labels.iloc[idx]['label'] 
            return image, label
        except Exception as e:
            print(f"Hiba a kép betöltésekor: {image_path}, hiba: {e}")
            return None

def create_data_loaders(images_folder, labels_file, img_size=(512, 512), batch_size=32):
    transform = transforms.Compose([
        transforms.Resize(img_size),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    dataset = ChessDataset(images_folder, labels_file, img_size, transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    images_folder = './images' 
    labels_file = './labels.xlsx'  
    train_loader, val_loader, test_loader = create_data_loaders(images_folder, labels_file)

    for images, labels in test_loader:
        print(f'Loading and preprocessing is done!')