from sympy import symbols, poly, expand,prod

# 定义符号
x = symbols('x')
x_i = symbols('x0:7')  # x0 到 x6
y_i = symbols('y0:7')  # 对应的 y 值

# 定义数据点，这里使用示例学号 20221161001
data_points = [(0, 1), (1, 1), (2, 6), (3, 1), (4, 0), (5, 0), (6, 1)]

# 提取 x 和 y 值
x_values = [point[0] for point in data_points]
y_values = [point[1] for point in data_points]

# Lagrange 插值多项式
L = sum(y_i * prod((x - x_j) / (x_i - x_j) for j, x_j in enumerate(x_values) if i != j) for i, (x_i, y_i) in enumerate(zip(x_values, y_values)))

# Newton 插值多项式
N = y_values[0]
for i in range(1, len(y_values)):
    diff = y_values[i] - poly(N, x).all_coeffs()[-1]
    N = poly(N, x) * (x - x_values[i-1]) + diff

# 将 Lagrange 和 Newton 插值多项式化为幂函数基的表达形式
L_power = expand(L)
N_power = expand(N)

# 打印结果
print("Lagrange 插值多项式 L_6(x):")
print(L_power)
print("\nNewton 插值多项式 N_6(x):")
print(N_power)