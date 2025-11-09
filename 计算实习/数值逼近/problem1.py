import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 图中正常显示负号

def bernstein_basis(i, n, x):
    """
    计算Bernstein基函数
    i: 基函数的索引
    n: 阶数
    x: 自变量值
    """
    return np.math.comb(n, i) * (x**i) * ((1-x)**(n-i))

def f(x):
    """
    目标函数 f(x) = (x^2 + 4x - 2)sin(πx)
    """
    return (x**2 + 4*x - 2) * np.sin(np.pi * x)

def bernstein_polynomial(f, n, x):
    """
    计算n阶Bernstein多项式
    """
    result = 0
    for i in range(n + 1):
        result += f(i/n) * bernstein_basis(i, n, x)
    return result

# 创建x值数组
x = np.linspace(0, 1, 1000)
plt.figure()
# 绘制四个3次Bernstein基函数
colors = ['red', 'blue', 'green', 'purple']
for i in range(4):
    y = [bernstein_basis(i, 3, xi) for xi in x]
    plt.plot(x, y, label=f'B$_{i}^3$(x)', color=colors[i])

plt.grid(True)
plt.legend()
plt.title('3次Bernstein基函数')
plt.xlabel('x')
plt.ylabel('y')


# 计算原函数值和不同阶数的Bernstein多项式值
y_original = [f(xi) for xi in x]
y_b4 = [bernstein_polynomial(f, 4, xi) for xi in x]
y_b10 = [bernstein_polynomial(f, 10, xi) for xi in x]
y_b20 = [bernstein_polynomial(f, 20, xi) for xi in x]

# 绘图
plt.figure()
plt.plot(x, y_original, 'red', label='f(x)', linewidth=2)
plt.plot(x, y_b4, 'blue', label='B$_4^f$(x)', linestyle='--')
plt.plot(x, y_b10, 'green', label='B$_{10}^f$(x)', linestyle='-.')
plt.plot(x, y_b20, 'purple', label='B$_{20}^f$(x)', linestyle=':')

plt.grid(True)
plt.legend(fontsize=12)
plt.title('原函数与不同阶Bernstein多项式逼近', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)

plt.show()

import matplotlib.pyplot as plt


plt.figure()
# 计算并绘制 n=4 的误差函数
y_true = [f(xi) for xi in x]
y_approx_4 = [bernstein_polynomial(f, 4, xi) for xi in x]
error_4 = [y_t - y_a for y_t, y_a in zip(y_true, y_approx_4)]
plt.plot(x, error_4, 'black', label='n=4的误差函数')

# 计算并绘制 n=10 的误差函数
y_true = [f(xi) for xi in x]
y_approx_10 = [bernstein_polynomial(f, 10, xi) for xi in x]
error_10 = [y_t - y_a for y_t, y_a in zip(y_true, y_approx_10)]
plt.plot(x, error_10, 'red', label='n=10的误差函数')

# 计算并绘制 n=20 的误差函数
y_approx_20 = [bernstein_polynomial(f, 20, xi) for xi in x]
error_20 = [y_t - y_a for y_t, y_a in zip(y_true, y_approx_20)]
plt.plot(x, error_20, 'blue', label='n=20的误差函数')

# 添加网格和图例
plt.grid(True)
plt.legend()

# 添加标题和轴标签
plt.title('Bernstein多项式逼近的误差函数对比')
plt.xlabel('x')
plt.ylabel('误差')

# 自动调整布局
plt.tight_layout()
plt.show()