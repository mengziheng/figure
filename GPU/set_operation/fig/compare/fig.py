import numpy as np
import matplotlib.pyplot as plt


# 定义函数
def func(x):
    return x - (x - 1)**32 * x / x**32


# 生成横坐标
x_values = [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# 计算纵坐标
y_values = [func(x) / func(x / 32) for x in x_values]

# 绘制曲线
plt.plot(x_values, y_values)

# # 设置对数坐标刻度
plt.xscale('log', base=2)

# 设置横坐标标签
# plt.xticks([2**i for i in range(1, 11)], ['2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])

# 设置图表标题和坐标轴标签
plt.title('E(memory transaction)')
plt.xlabel('x')
plt.ylabel('y')

# 保存图表为PDF文件
plt.savefig('function_plot.pdf')

# 显示图表
plt.show()
