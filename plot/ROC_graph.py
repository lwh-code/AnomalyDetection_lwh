import matplotlib.pyplot as plt
from pylab import *




if __name__ == '__main__':

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)

    #中文无法显示进行配置
    # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（解决中文无法显示的问题）
    # mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号“-”显示方块的问题
   # mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体（解决中文无法显示的问题）

    #这是设置x轴和y轴标签的方式
    plt.xlabel('FPRate')
    plt.ylabel('TPRate')
    plt.title('ROC曲线示意图')

    #设置坐标的范围
    # plt.xlim((0, 1))
    # plt.ylim((0, 1))

    #学习如何进行点之间的连线
    plt.plot([0, 0, 1/3, 2/3, 2/3, 1], [0, 1/2, 1/2, 1/2, 1, 1], color='#00FFFF')
    plt.scatter([0, 1/3, 2/3, 2/3, 1], [1/2, 1/2, 1/2, 1, 1], color='#00FFFF')
    plt.text(0, 1/2, 'gate=0.9')
    plt.text(1/3, 1/2, 'gate=0.7')
    plt.text(2/3, 1/2, 'gate=0.5')
    plt.text(2/3, 1, 'gate=0.4')
    plt.text(1, 1, 'gate=0.2')

    #这里进行图片的保存
    plt.title('ROC曲线图')
    plt.savefig('D:\\桌面\\AUROC_example.png', dpi=600, bbox_inches='tight')

    plt.show()