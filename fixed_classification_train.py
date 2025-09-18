#!/usr/bin/env python3
"""
Fixed YOLOv5 classification training script
Bypasses the problematic dataloader
"""

import os
import sys
import torch
import torchvision
from pathlib import Path

# Add yolov5original to path
sys.path.append('yolov5original')

def fixed_classification_train():
    """Fixed classification training"""
    
    print("🔧 Fixed YOLOv5 Classification Training...")
    
    # Check dataset
    dataset_path = Path("yolov5original/datasets/regurgitationV1-cls")
    if not dataset_path.exists():
        print("❌ Dataset not found")
        return
    
    print("✅ Dataset found")
    
    # Create simple dataloader using torchvision
    train_path = dataset_path / "train"
    val_path = dataset_path / "val"
    
    if not train_path.exists():
        print("❌ Train directory not found")
        return
    
    # Create transforms
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((416, 416)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    train_dataset = torchvision.datasets.ImageFolder(str(train_path), transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(str(val_path), transform=transform) if val_path.exists() else None
    
    print(f"✅ Train dataset: {len(train_dataset)} images, {len(train_dataset.classes)} classes")
    if val_dataset:
        print(f"✅ Val dataset: {len(val_dataset)} images")
    
    # Create dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=4, 
        shuffle=False, 
        num_workers=0
    ) if val_dataset else None
    
    # Load model
    try:
        from models.yolo import ClassificationModel
        model = ClassificationModel('yolov5s-cls.pt', nc=3)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Training setup
    device = torch.device('cpu')
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (1 epoch)
    print("🚀 Starting training...")
    model.train()
    
    for epoch in range(1):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Accuracy: {100.*correct/total:.2f}%')
        
        print(f'Epoch {epoch} completed - Average Loss: {total_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%')
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    fixed_classification_train()

