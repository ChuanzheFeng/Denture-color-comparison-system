# 导入库
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image

# PyTorch相关库
import torch
import torchvision.transforms as transforms
import torchvision.models as torch_models

import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

# 定义参数
IMG_SIZE_LENGTH = 224
IMG_SIZE_WIDTH = 224
IMG_CHANNELS = 3

# 定义类别名称
names = ["良好", "较浅", "较深", "深"]

# 加载TF模型的函数
def load_tf_model(architecture_path, weights_path):
    with open(architecture_path, "r") as json_file:
        loaded_model_json = json_file.read()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
    return loaded_model

# 加载所有TensorFlow模型
models = {
    "CNN": load_tf_model("model/model_CNN_architecture.json", "model/model_CNN_weights.h5"),
    "Inception": load_tf_model("model/model_inc_architecture.json", "model/model_inc_weights.h5"),
    "TF_ResNet": load_tf_model("model/model_resnet_architecture.json", "model/model_resnet_weights.h5"),
    "VGG": load_tf_model("model/model_vgg_architecture.json", "model/model_vgg_weights.h5")
}

# 创建一个自定义的ResNet模型，输出尺寸为4
def custom_resnet50(output_size):
    model = torch_models.resnet50(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, output_size)  # 将全连接层的输出尺寸设置为4
    return model


# 加载PyTorch模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_pytorch = custom_resnet50(output_size=4)  # 使用重命名的模块名
resnet_pytorch.load_state_dict(torch.load("model/model_resnet_pytorch.pth", map_location=device))
resnet_pytorch = resnet_pytorch.to(device).eval()
models["PyTorch_ResNet"] = resnet_pytorch

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_LENGTH, IMG_SIZE_WIDTH)),
    transforms.ToTensor(),
])

def predict_image(img_path, model, is_pytorch=False):
    # 打开图片并将其转换为RGB模式
    img = Image.open(img_path).convert("RGB")
    # 将图片调整为指定的大小
    img = img.resize((IMG_SIZE_LENGTH, IMG_SIZE_WIDTH))
    # 将图片转换为NumPy数组
    img_np = np.array(img)
    assert img_np.shape == (IMG_SIZE_LENGTH, IMG_SIZE_WIDTH, IMG_CHANNELS)  # 请根据您的具体需求修改

    if is_pytorch:
        # 将NumPy数组转换为PyTorch张量
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0  # 为了与ToTensor一致，除以255
        img_tensor = img_tensor.unsqueeze(0).to(device)  # 添加批次维度并移动到设备上
        with torch.no_grad():
            output = model(img_tensor)
        predictions = torch.nn.functional.softmax(output, dim=1)
        return predictions.cpu().numpy()
    else:
        img_np = img_np.astype(np.float32) / 255.0  # 与ToTensor一致，除以255
        img_np = np.expand_dims(img_np, axis=0)
        return model.predict(img_np)

def get_image_path():
    file_path = filedialog.askopenfilename(title="选择图片", filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")])
    return file_path

def predict_and_display():
    img_path = get_image_path()
    if img_path:
        results_text = "预测结果:\n\n"
        for model_name, model in models.items():
            is_pytorch_model = model_name == "PyTorch_ResNet"
            predictions = predict_image(img_path, model, is_pytorch=is_pytorch_model)
            pred_probs = predictions[0]
            results_text += f"模型{model_name}预测的此牙色泽：\n"
            for i, name in enumerate(names):
                results_text += f"{name}的概率分别为: {pred_probs[i]:.4f}\n"
            results_text += "\n"
        messagebox.showinfo("预测结果", results_text)

root = tk.Tk()
root.title("牙色泽预测器")
root.geometry("300x200")

frame = tk.Frame(root)
frame.pack(pady=60)

btn_upload = tk.Button(frame, text="选择图片进行预测", command=predict_and_display)
btn_upload.pack()

root.mainloop()