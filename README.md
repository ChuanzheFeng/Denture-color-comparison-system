# Denture-color-comparison-system
Enter a photo of dentures, and the system can compare it with standard teeth and get the result of comparing the color of the dentures with the color of standard teeth.
## Photo_process.py ##
调整图片的分辨率并选择一定的角度。
## Photo_process2.py ##
将牙从原图片中抠出来即换成纯黑背景
## PyTorch.py ##
查看现在的PyTorch是CPU还是GPU版本
## resnet.py ##
使用Resnet对图片进行分类：
0.3--较浅
0.5--良好
0.7--较深
0.9--深
## testCNN.ipynb ##
使用CNN、VGG、Inception、tf_ResNet、PyTorch_ResNet对图片进行分类：
0--良好
1--较浅
2--较深
3--深
## testCNN.py ##
加载CNN、VGG、Inception、tf_ResNet、PyTorch_ResNet模型并对输入的图片的牙齿色泽进行预测
## testRes.py ##
加载ResNet模型并对输入的牙齿图片进行分类
## testRes_cpu.py ##
上一文件的CPU版本
