import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy

def train_model():
    # 1. Image Augmentation Matrix Setup
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data'
    
    # Structural directory validations
    if not os.path.exists(os.path.join(data_dir, 'train')):
        print("⚠️ Error: Please place your 'train' and 'val' folders inside the 'data/' directory first!")
        return

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=0) for x in ['train', 'val']}
    
    class_names = image_datasets['train'].classes
    print(f"🎯 Detected Classes: {class_names}")

    # 2. Compile Framework Backbone (ResNet18)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Execution Device: {device}")
    
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Freeze initial convolutional layers
    for param in model.parameters():
        param.requires_grad = False

    # Swap outer layer parameters
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    # 3. Training/Validation Phase with Checkpoint Tracking
    epochs = 5
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'\n--- Epoch {epoch+1}/{epochs} ---')
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy model checkpoint only if verification returns optimization spikes
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f'\n🏆 Training Completed. Best Val Accuracy: {best_acc:.4f}')

    # Save highest performing metrics mapping block
    model.load_state_dict(best_model_wts)
    torch.save({'state_dict': model.state_dict(), 'classes': class_names}, 'plant_model.pth')
    print("💾 Production checkpoints securely saved as 'plant_model.pth'")

if __name__ == '__main__':
    train_model()