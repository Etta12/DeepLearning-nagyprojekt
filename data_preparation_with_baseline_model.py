import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import wandb
from PIL import Image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torchmetrics
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ChessDataset(Dataset):
    def __init__(self, images_folder, labels_file, img_size, transform=None):
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
            label = label - 1
            return image, label
        except Exception as e:
            print(f"Hiba a kép betöltésekor: {image_path}, hiba: {e}")
            return None

def create_data_loaders(images_folder, labels_file, img_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(img_size),  
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
    ])

    torch.manual_seed(seed)
    dataset = ChessDataset(images_folder, labels_file, img_size, transform)

    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

class ImageClassifier(pl.LightningModule):
    def __init__(self, model, num_classes, learning_rate):
        super().__init__()
        self.model = model
        self.accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.learning_rate = learning_rate
        self.test_predictions = []
        self.test_labels = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        pred_labels = self(images)

        loss = F.cross_entropy(pred_labels, labels)
        accuracy = self.accuracy(pred_labels, labels)

        self.log("train_acc", accuracy, on_epoch=True)
        self.log("train_acc", accuracy, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        pred_labels = self(images)

        loss = F.cross_entropy(pred_labels, labels)
        accuracy = self.accuracy(pred_labels, labels)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        images, labels = batch
        pred_labels = self(images)

        loss = F.cross_entropy(pred_labels, labels)
        accuracy = self.accuracy(pred_labels, labels)

        self.log("test_loss", loss)
        self.log("test_acc", accuracy)

        self.test_predictions.append(pred_labels) 
        self.test_labels.append(labels)

        return loss

    def on_test_epoch_end(self):
        self.test_predictions = torch.cat(self.test_predictions).cpu().float().numpy() 
        self.test_labels = torch.cat(self.test_labels).cpu().numpy() 

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_test_predictions(self):
        return self.test_predictions, self.test_labels

if __name__ == "__main__":
    
    wandb.login(key='cb287032b3b9fe50af7194d915f55c0af69047f3')
    
    # SEED
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # HYPERPARAMETERS    
    num_classes = 3
    img_size = (512, 512)
    batch_size = 10
    num_epochs = 1
    learning_rate = 0.001
    width = 64
     
    # DATA LOADING AND PREPARATION
    images_folder = './images' 
    labels_file = './labels.xlsx'  
    train_loader, val_loader, test_loader = create_data_loaders(images_folder, labels_file, img_size, batch_size)

    print(f'Data loading and data preparation is done!')
     
    # MODELING    
    model = torch.nn.Sequential(
    torch.nn.Conv2d(in_channels=3, out_channels=width, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, stride=2, padding=1),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(output_size=1),
    torch.nn.Flatten(),
    torch.nn.Linear(in_features=width, out_features=num_classes),
    )

    lit_model = ImageClassifier(model=model, num_classes=num_classes, learning_rate=learning_rate)

    wandb_logger = pl.loggers.WandbLogger(project="chess_project", log_model="all")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor="val_acc", mode="max")
    early_stopping_callback = EarlyStopping(
        monitor='val_acc',
        patience=3,
        mode='max',
        verbose=True
    )

    trainer = pl.Trainer(
        log_every_n_steps=1,
        max_epochs=num_epochs,
        precision="16-mixed",
        accelerator="cpu",
        devices=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )

    # TRAIN THE MODEL
    train_loader, val_loader, test_loader = create_data_loaders(images_folder, labels_file, img_size, batch_size)
    trainer.fit(lit_model, train_loader, val_loader)
    print('Model training is done!')
    
    # TEST THE MODEL
    trainer.test(lit_model, test_loader)
    print('Model testing is done!')
    wandb.finish()
    
    # PREDICTIONS
    predictions, true_labels = lit_model.get_test_predictions()
    probabilities = F.softmax(torch.tensor(predictions), dim=1)
    predicted_classes = torch.argmax(probabilities, dim=1)
    
    # TEST ACCURACY AND CONFUSION MATRIX
    cm = confusion_matrix(true_labels, predicted_classes)
    correct_predictions = cm.diagonal().sum()
    total_predictions = cm.sum()
    test_accuracy = correct_predictions / total_predictions
 
    print(f'The accuracy on the test set is {test_accuracy}.')
    print('The confusion matrix:')
    print(cm)