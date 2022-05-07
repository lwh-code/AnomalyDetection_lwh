import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

def get_Mnist():
    filename = 'D:\\桌面\\test.jpg'
    # 读取文件
    # Data set
    """加载Mnist手写数据集训练集"""
    mnist_train = datasets.MNIST('D:\\桌面\\data_mnist',
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([transforms.ToTensor()]))
    """加载Mnist手写数据集测试集合"""
    mnist_test = datasets.MNIST('D:\\桌面\\data_mnist',
                                train=False,
                                download=True,
                                transform=transforms.Compose([transforms.ToTensor()]))

    # Data loader
    train_data = torch.utils.data.DataLoader(mnist_train, batch_size=1, shuffle=True)
    test_data = torch.utils.data.DataLoader(mnist_test, batch_size=1, shuffle=True)
    print(len(train_data))
    print(len(test_data))

    os.mkdir('images')
    for i in range(10):
        data, target = next(iter(test_data))  # 迭代器
        new_data = data[0][0].clone().numpy()  # 拷贝数据
        plt.imsave('images/' + str(i) + str(target) + '.png', new_data)
        print(target)

    # img_matrix = plt.imread(filename)
    # print(img_matrix)
    # 将文件转化为矩阵，并通过第一个参数的路径进行保存
    # matplotlib.image.imsave('D:\\桌面\\mypic.png', img_matrix)
    # plt.imshow(img_matrix, cmap='gray')
    # plt.show()

def show_binary():
    #加载图片
    image = Image.open('images/num_8.png')
    #这是转化为灰度图像
    # image.convert('L')
    print(image)
    #对图片矩阵进行遍历
    binary_matrix=np.zeros((image.height,image.width))
    for i in range(image.height):  # 转化为二值矩阵
        for j in range(image.width):
            pixel = image.get_pixel(i, j)
            if pixel.red != 0:
                binary_matrix[i, j] = 1
            else:
                binary_matrix[i, j] = 0
    print(binary_matrix)


if __name__ == '__main__':
    show_binary()