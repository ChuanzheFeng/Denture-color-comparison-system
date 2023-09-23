import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import transforms
from PIL import Image
import tkinter as tk
from tkinter import filedialog

#model = resnet50(pretrained=True)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_features = model.fc.in_features
num_classes = 1  
model.fc = nn.Linear(num_features, num_classes)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

tooth_test_dataset = datasets.ImageFolder(root="base", transform=transform)
tooth_test_dataloader = DataLoader(dataset=tooth_test_dataset)

def browse_model_file():
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
    if model_path:
        model_path_var.set(model_path)

def load_and_calculate_baseline():
    model_path = model_path_var.get()
    try:
        model.load_state_dict(torch.load(model_path))
        model_status_var.set("模型已加载，正在计算基线值，请稍候")
        
        total = 0
        for images, labels in tooth_test_dataloader:
            model.eval()
            outputs = model(images)
            probabilities = outputs.item()
            total = total + probabilities
        baseline = total / len(tooth_test_dataset)
        baseline_var.set(f"标准牙的测试结果是: {baseline}")
        model_status_var.set("模型已加载，基线值已计算")
        
    except Exception as e:
        model_status_var.set("模型加载失败：" + str(e))
        baseline_var.set("")

def evaluate_image():
    data_dir = file_path_var.get()
    image = Image.open(data_dir)
    image = transform(image)
    image = image.unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    res = output.item()
    if res < float(baseline_var.get().split(": ")[1]) - 0.1:
        res_msg = "目标牙偏浅"
    elif res > float(baseline_var.get().split(": ")[1]) + 0.1:
        res_msg = "目标牙偏深"
    else:
        res_msg = "目标牙颜色良好"
    result_var.set("检测结果：" + res_msg + f" (res值: {res})")

def browse_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        file_path_var.set(file_path)

root = tk.Tk()
root.title("牙齿颜色检测")

model_path_var = tk.StringVar()
tk.Entry(root, textvariable=model_path_var, width=50).pack(pady=10)
tk.Button(root, text="浏览模型文件", command=browse_model_file).pack(pady=10)
tk.Button(root, text="加载模型并计算基线值", command=load_and_calculate_baseline).pack(pady=10)

model_status_var = tk.StringVar()
tk.Label(root, textvariable=model_status_var, font=("Arial", 12)).pack(pady=10)

baseline_var = tk.StringVar()
tk.Label(root, textvariable=baseline_var, font=("Arial", 12)).pack(pady=10)

file_path_var = tk.StringVar()
tk.Entry(root, textvariable=file_path_var, width=50).pack(pady=10)
tk.Button(root, text="浏览", command=browse_file).pack(pady=10)
tk.Button(root, text="检测", command=evaluate_image).pack(pady=10)

result_var = tk.StringVar()
tk.Label(root, textvariable=result_var, font=("Arial", 12)).pack(pady=10)

root.mainloop()