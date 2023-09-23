import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
from torchvision.models import resnet50, ResNet50_Weights

device = torch.device("cuda")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class ModifiedImageFolder(ImageFolder):
    def __getitem__(self, index):
        # 调用父类的 __getitem__ 方法获取原始数据
        original_data = super(ModifiedImageFolder, self).__getitem__(index)
        # 修改标签为你想要的值
        if original_data[1] == 0:
            modified_data = (original_data[0], 0.3)  # 这里将标签设置为 0.3
        elif original_data[1] == 1:
            modified_data = (original_data[0], 0.5)  # 这里将标签设置为 0.5
        elif original_data[1] == 2:
            modified_data = (original_data[0], 0.7)  # 这里将标签设置为 0.7
        elif original_data[1] == 3:
            modified_data = (original_data[0], 0.9)  # 这里将标签设置为 0.9
        else:
            modified_data = (original_data[0], 1.0)  

        return modified_data

# 训练集
tooth_dataset = ModifiedImageFolder(root="data\\resnet\\",
                                     transform=transform)
tooth_dataloader = DataLoader(dataset=tooth_dataset, batch_size=32, shuffle=True)

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

num_features = model.fc.in_features

num_classes = 1
model.fc = nn.Linear(num_features, num_classes)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
best_loss = 1e100
model.to(device)
num_epochs = 30
losses = []
test_accuracy = []
for epoch in range(num_epochs):
    model.train()
    trcorrect = 0
    trtotal = 0
    running_loss = 0.0
    for images, labels in tooth_dataloader:
        images = images.to(device)
        labels = torch.Tensor(labels).to(device)
        labels = labels.float()
        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.unsqueeze(1)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, trpredicted = outputs.max(1)
        trtotal += labels.size(0)
        trcorrect += trpredicted.eq(labels).sum().item()
        losses.append(running_loss / len(tooth_dataloader))
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(tooth_dataloader)}")
    loss = running_loss / len(tooth_dataloader)
    if loss < best_loss:
        torch.save(model.state_dict(), 'model/model_resnetwithAdam.pth')
    total_samples = 0

