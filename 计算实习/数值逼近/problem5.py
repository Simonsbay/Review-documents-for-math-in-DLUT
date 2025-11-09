import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['SimHei']  # 图中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 图中正常显示负号
from mpl_toolkits.mplot3d import Axes3D

# 定义三次 Bernstein 基函数
def bernstein_poly(i, n, t):
    return np.math.comb(n, i) * (t**i) * ((1-t)**(n-i))

# 定义三次 Bézier 曲线的点
def bezier_point(points, t):
    n = len(points) - 1
    return sum(bernstein_poly(i, n, t) * points[i] for i in range(n+1))

# 控制点
p0 = np.array([0, 0])
p1 = np.array([1, -1])
p2 = np.array([2, 1])
p3 = np.array([3, 0])

# 计算曲线上的点
t_values = np.linspace(0, 1, 100)
curve_points = [bezier_point([p0, p1, p2, p3], t) for t in t_values]

# 绘制控制多边形
plt.plot([p0[0], p1[0], p2[0], p3[0]], [p0[1], p1[1], p2[1], p3[1]], 'ro-', label='控制多边形')

# 绘制 Bézier 曲线
x_curve, y_curve = zip(*curve_points)  # 将列表解包为两个元组
plt.plot(x_curve, y_curve, 'b-', label='Bezier曲线')

# 标记控制点
for point, label in zip([p0, p1, p2, p3], ['P$_0$', 'P$_1$', 'P$_2$', 'P$_3$']):
    plt.scatter(point[0], point[1], color='red')
    plt.text(point[0], point[1], label, fontsize=12, ha='right')


# 设置图例和标签
plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('三次Bezier曲线和控制多边形')
plt.grid(True)
plt.axis('equal')  # 保持比例
plt.tight_layout()
plt.show()


# 定义 Bézier 曲面上的点
def bezier_surface_point(points, u, v):
    n = len(points) - 1
    m = len(points[0]) - 1
    return sum(sum(bernstein_poly(i, n, u) * bernstein_poly(j, m, v) * points[i][j]
                   for j in range(m+1)) for i in range(n+1))

# 控制点
P = np.array([
    [(0, 0, 1), (1, 0, 2), (2, 0, 1)],
    [(0, 1, 2), (1, 1, 2), (2, 1, 2)],
    [(0, 2, 1), (1, 2, 2), (2, 2, 1)]
])

# 创建网格
u_values = np.linspace(0, 1, 10)
v_values = np.linspace(0, 1, 10)
U, V = np.meshgrid(u_values, v_values)

# 计算曲面上的点
X, Y, Z = [], [], []
for u in u_values:
    for v in v_values:
        x, y, z = bezier_surface_point(P, u, v)
        X.append(x)
        Y.append(y)
        Z.append(z)


# 将列表转换为 NumPy 数组
X, Y, Z = np.array(X), np.array(Y), np.array(Z)
X, Y, Z = X.reshape(U.shape), Y.reshape(U.shape), Z.reshape(U.shape)

# 绘制曲面
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6)

# 绘制控制点
for i in range(3):
    for j in range(3):
        ax.scatter(P[i, j, 0], P[i, j, 1], P[i, j, 2], color='red', s=50)
        ax.text(P[i, j, 0], P[i, j, 1], P[i, j, 2], f'P{i},{j}', color='black')
# 绘制控制多边形
# u方向控制曲线
for j in range(3):
    ax.plot([P[0, j, 0], P[1, j, 0], P[2, j, 0]],
            [P[0, j, 1], P[1, j, 1], P[2, j, 1]],
            [P[0, j, 2], P[1, j, 2], P[2, j, 2]], 'r')

# v方向控制曲线
for i in range(3):
    ax.plot([P[i, 0, 0], P[i, 1, 0], P[i, 2, 0]],
            [P[i, 0, 1], P[i, 1, 1], P[i, 2, 1]],
            [P[i, 0, 2], P[i, 1, 2], P[i, 2, 2]], 'r')

# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('2x2 Bezier曲面')
plt.tight_layout()
plt.show()
