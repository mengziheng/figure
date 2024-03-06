import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def func(x):
    return x - (x - 1) ** 32 * x / x ** 32 

# 生成横坐标
x_values = range(1,1025)
    
# 计算纵坐标
y_values = [func(x) for x in x_values]

# 绘制曲线
plt.plot(x_values, y_values)

plt.yticks(range(0, 33, 2))  # 设置 x 轴刻度
# plt.yticks([2, 4, 6, 8, 10, 12])  # 设置 y 轴刻度

# # 设置对数坐标刻度
plt.xscale('log', base=2)

# 设置横坐标标签
plt.xticks([2**i for i in range(1, 11)], ['2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])

# 设置图表标题和坐标轴标签
plt.title('E(memory transaction)')
plt.xlabel('x')
plt.ylabel('y')

plt.ylim(0)

# plt.grid(True)

def addpoint(point_x):
    point_y = func(point_x)
    # 在指定点处添加标记
    plt.scatter(point_x, point_y, color='black', label='Point')
    # # 绘制平行于x轴的虚线
    # plt.axhline(y=point_y, color='gray', linestyle='--')
    # # 绘制平行于y轴的虚线
    # plt.axvline(x=point_x, color='gray', linestyle='--')
    # 绘制平行于x轴的虚线
    plt.plot([0, point_x], [point_y, point_y], color='gray', linestyle='--')

    # 绘制平行于y轴的虚线
    plt.plot([point_x, point_x], [0, point_y], color='gray', linestyle='--')

    
addpoint(256)

addpoint(64)

addpoint(16)

# 保存图表为PDF文件
plt.savefig('function_plot.pdf')

# 显示图表
plt.show()
