import os
from PIL import Image

# 输入文件夹路径
input_folder = "data/0919"
# 输出文件夹路径
output_folder = "data/processed_0919"

# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)

# 旋转图片一周，每次旋转90度
rotation_step = 90

# 处理每个日期文件夹
for date_folder in ["13", "14", "15", "16"]:
    input_date_folder = os.path.join(input_folder, date_folder)
    output_date_folder = os.path.join(output_folder, date_folder)

    # 创建输出日期文件夹
    os.makedirs(output_date_folder, exist_ok=True)

    # 获取日期文件夹下的所有图片文件
    image_files = [f for f in os.listdir(input_date_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # 处理每张图片
    for image_file in image_files:
        input_image_path = os.path.join(input_date_folder, image_file)
        output_image_path = os.path.join(output_date_folder, image_file)

        # 打开图片
        image = Image.open(input_image_path)

        # 旋转图片一周
        for rotation_deg in range(0, 360, rotation_step):
            
            # 放缩图片分辨率为224x224
            resized_image = image.resize((224, 224))
            
            # 旋转图片
            rotated_image = resized_image.rotate(rotation_deg)

            # 保存处理后的图片
            output_filename = f"rotated_{rotation_deg}_degrees_{image_file}"
            output_image_path = os.path.join(output_date_folder, output_filename)
            rotated_image.save(output_image_path)

print("图片处理完成并保存至目标文件夹。")