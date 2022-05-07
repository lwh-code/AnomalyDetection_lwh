import torch
import matplotlib
import matplotlib.pyplot as plt


#这是绘制数据沿着x轴分布的图像
def plot_1():
    x = torch.randn(300) * 2
    y = torch.randn(300)

    # 绘制以0，0为原点的直角坐标体系
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')

    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)

    ax.scatter(x, y, c='#00ffff', s=3)

    ax.scatter([0, 6], [6, 0], c='r', s=18)
    ax.text(0.2, 6, 'A', size=15)
    ax.text(6.2, 0.2, 'B', size=15)

    #f.suptitle("沿x轴数据分布图", size='x-large', weight='bold')
    save(f, 'x轴数据分布图')
    f.show()

#绘制沿着y=x这一直线分布的图像
def plot_2():
    # 先从torch中获取正态分布的点，先获取一批参数
    x = torch.randn(300) * 3
    y = x + torch.randn(300)


    # 绘制以0，0为原点的直角坐标体系
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')

    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)

    ax.scatter(x, y, c='#00ffff', s=3)

    ax.scatter([-5, 5], [5, 5], c='r', s=18)
    ax.text(-5, 5, 'A', size=15)
    ax.text(5, 5, 'B', size=15)


    #f.suptitle("沿y=x数据分布图", size='x-large', weight='bold')
    save(f, '关于y=x分布的数据图')
    f.show()

#绘制旋转缩放之后的图片
def plot_3():
    x = torch.randn(300)
    y = torch.randn(300)

    # 绘制以0，0为原点的直角坐标体系
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')

    ax.set_ylim(-8, 8)
    ax.set_xlim(-8, 8)

    ax.scatter(x, y, c='#00ffff', s=3)

    ax.scatter([0, 3], [6, 0], c='r', s=18)
    ax.text(0.2, 6, 'A', size=15)
    ax.text(3.2, 0.2, 'B', size=15)

    #f.suptitle("按照PCA旋转缩放之后的数据分布图", size='x-large', weight='bold')
    save(f, '按照PCA旋转缩放之后的数据分布')
    f.show()

#将图片进行保存方法
def save(instance, name:str):
    instance.savefig('D:\\桌面\\'+name, dpi=600, bbox_inches='tight')

def test():
    x = torch.randn(300) * 12
    y = x + torch.randn(300)
    plt.scatter(x, y, color='#00FFFF',s=5)  #设置点的大小

    # plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.axis([-6, 6, -6, 6])
    plt.title('数据分布图')
    plt.show()

#将在本方法中进行散点图的绘制
if __name__ == '__main__':
    plot_1()
    plot_2()
    plot_3()