import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from PIL import Image


# 测试集
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
tooth_test_dataset = datasets.ImageFolder(root="base", transform=transform)
tooth_test_dataloader = DataLoader(dataset=tooth_test_dataset)
model = resnet50(pretrained=True)
num_features = model.fc.in_features
num_classes = 1
model.fc = nn.Linear(num_features, num_classes)
model.load_state_dict(torch.load('resnetwithAdam111.pth',
                      map_location=torch.device('cpu')))
total = 0
likely = []
probality = []
print("初始化数据中，请稍候")
for images, labels in tooth_test_dataloader:
    model.eval()
    likelys = 0
    outputs = model(images)
    probabilities = outputs.item()
    probality.append(probabilities)
    #     probality.append(probabilities)
    #     if probabilities > standard:
    #         likelys = standard / probabilities
    #     else:
    #         likelys = probabilities / standard
    total = total + probabilities
#     likely.append(likelys)
baseline = total / len(tooth_test_dataset)
print("标准牙的测试结果是:", baseline)

while True:
    data_name = input("请输入要检测的牙齿的图片名称(输入exit退出):")
    if data_name == "exit":
        break
    data_dir = "test//" + data_name

    # 打开并预处理图片
    image = Image.open(data_dir)
    image = transform(image)

    # 将图片扩展为一个批次的张量（因为模型通常要求输入为批次）
    image = image.unsqueeze(0)

    total1 = 0
    with torch.no_grad():
        output = model(image)

    res = output.item()
    print(res)
    if res < baseline-0.1:
        print("目标牙偏浅")
    elif res > baseline+0.1:
        print("目标牙偏深")
    else:
        print("目标牙颜色良好")
