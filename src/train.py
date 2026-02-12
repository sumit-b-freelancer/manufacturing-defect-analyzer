import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import argparse

from src.models.model import SentinelModel
from src.dataset import DefectDataset, get_transforms

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        # Mixed Precision Context
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Scaler logic for backprop
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_description(f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Hyperparameters
    BATCH_SIZE = args.batch_size
    LR = args.lr
    EPOCHS = args.epochs
    NUM_CLASSES = args.num_classes
    MODEL_NAME = args.model_name
    DATA_DIR = args.data_dir

    # Data
    train_dataset = DefectDataset(os.path.join(DATA_DIR, 'train'), transform=get_transforms(is_train=True))
    val_dataset = DefectDataset(os.path.join(DATA_DIR, 'val'), transform=get_transforms(is_train=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    # Use pretrained=True for Transfer Learning
    print(f"Loading {MODEL_NAME} with ImageNet weights...")
    model = SentinelModel(model_name=MODEL_NAME, num_classes=NUM_CLASSES, pretrained=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    # Lower learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=1e-4) 
    scaler = GradScaler() # For FP16

    best_acc = 0.0

    print("Starting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        if val_acc >= best_acc:
            best_acc = val_acc
            # Save to root directory
            save_path = os.path.join(os.getcwd(), "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='tf_efficientnetv2_s')
    args = parser.parse_args()
    
    # Create dummy dirs if not exist for testing purposes logic would be here or external
    # But main assumes they exist.
    if os.path.exists(args.data_dir):
        main(args)
    else:
        print(f"Dataset path {args.data_dir} does not exist.")
