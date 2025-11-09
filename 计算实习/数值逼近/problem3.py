import numpy as np
import matplotlib.pyplot as plt
from sympy import Symbol, expand, simplify, N

# 设置matplotlib配置
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 图中正常显示负号

# 定义Lagrange基函数
def lagrange_basis(x, i, x_points):
    term = 1
    for j in range(len(x_points)):
        if j != i:
            term *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return term

# 定义Lagrange插值多项式
def lagrange_interpolation(x_points, y_points, x):
    return sum(y_points[i] * lagrange_basis(x, i, x_points) for i in range(len(x_points)))

# 定义Newton差商
def divided_diff(x_points, y_points):
    n = len(y_points)
    coef = np.copy(y_points).astype(float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x_points[i] - x_points[i-j])
    return coef

# 定义Newton插值多项式
def newton_interpolation(coef, x_points, x):
    n = len(coef) - 1
    p = coef[n]
    for k in range(1, n+1):
        p = coef[n-k] + (x - x_points[n-k]) * p
    return p

# 数据点
x_points = np.array([0, 1, 2, 3, 4, 5, 6])
y_points = np.array([1, 1, 6, 1, 1, 5, 5])

# 计算Newton差商
coef = divided_diff(x_points, y_points)

# 绘制Lagrange基函数
x_vals = np.linspace(0, 6, 1000)
plt.figure()
for i in range(len(x_points)):
    l = lagrange_interpolation(x_points, [0]*i + [1] + [0]*(len(x_points)-i-1), x_vals)
    plt.plot(x_vals, l, label=f'l$_{i}$')
# 添加图例
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="拉格朗日插值基函数")
# 添加标题和轴标签
plt.title('拉格朗日基函数')
plt.xlabel('x')
plt.ylabel('l$_i$(x)')
# 显示网格
plt.grid(True)
# 自动调整布局
plt.tight_layout()
plt.show()

# 定义Lagrange基函数（符号计算版本）
def lagrange_basis_symbolic(i, x_points):
    x = Symbol('x')
    term = 1
    for j in range(len(x_points)):
        if j != i:
            term *= (x - x_points[j]) / (x_points[i] - x_points[j])
    return expand(term)

# 计算Lagrange插值多项式的符号表达式
def lagrange_interpolation_symbolic(x_points, y_points):
    x = Symbol('x')
    polynomial = 0
    for i in range(len(x_points)):
        polynomial += y_points[i] * lagrange_basis_symbolic(i, x_points)
    return expand(polynomial)

# 计算Lagrange插值多项式
L6 = lagrange_interpolation_symbolic(x_points, y_points)
# 将Lagrange插值多项式的系数转换为浮点数并保留六位小数
L6 = N(L6, 6)
print("Lagrange插值多项式L₆(x)的幂函数基表达式：")
print(f"L₆(x) = {L6}")

# 计算Newton插值多项式的符号表达式
def newton_interpolation_symbolic(x_points, y_points):
    x = Symbol('x')
    coef = divided_diff(x_points, y_points)
    n = len(coef)
    polynomial = coef[0]
    term = 1
    for i in range(1, n):
        term *= (x - x_points[i-1])
        polynomial += coef[i] * term
    return expand(polynomial)

# 计算Newton插值多项式
N6 = newton_interpolation_symbolic(x_points, y_points)
N6 = N(N6, 6)
print("\nNewton插值多项式N₆(x)的幂函数基表达式：")
print(f"N₆(x) = {N6}")

# 验证两个多项式是否相等
diff = simplify(L6 - N6)
print(f"\n两个多项式的差：{diff}")

# 计算Lagrange插值多项式在一系列x值上的值
x_lagrange = np.linspace(0, 6, 1000)
y_lagrange = lagrange_interpolation(x_points, y_points, x_lagrange)
# 绘制Lagrange插值多项式和数据点
plt.figure()
plt.plot(x_lagrange,y_lagrange,label='L$_6$(x)')
plt.scatter(x_points, y_points, color='red', label='数据点')
plt.title('数据点集{x$_i$,y$_i$}与L$_6$(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
# 自动调整布局
plt.tight_layout()
plt.show()

