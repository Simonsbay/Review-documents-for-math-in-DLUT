import math
import numpy as np
def f(x):
    """被积函数 f(x) = 2/(1+x^2)"""
    return 2 / (1 + x * x)


def linspace(start, end, num):
    if num < 2:
        return [start]
    step = (end - start) / (num - 1)
    return [start + step * i for i in range(num)]


def composite_trapezoidal(f, a, b, n):
    """复化梯形公式
    参数:
        f: 被积函数
        a, b: 积分区间端点
        n: 区间等分数
    """
    h = (b - a) / n
    x = linspace(a, b, n + 1)
    result = f(x[0]) / 2.0 + f(x[-1]) / 2.0
    for i in range(1, n):
        result += f(x[i])
    return h * result


def composite_simpson(f, a, b, n):
    """复化Simpson公式
    参数:
        f: 被积函数
        a, b: 积分区间端点
        n: 区间等分数(须为偶数)
    """
    if n % 2 != 0:
        n += 1
    h = (b - a) / n
    x = linspace(a, b, n + 1)

    result = f(x[0]) + f(x[-1])
    for i in range(1, n, 2):
        result += 4 * f(x[i])
    for i in range(2, n - 1, 2):
        result += 2 * f(x[i])

    return h / 3 * result


def composite_cotes(f, a, b, n):
    """复化Cotes公式
    参数:
        f: 被积函数
        a, b: 积分区间端点
        n: 区间等分数
    """
    if n % 2 == 1:
        raise ValueError("n 必须是偶数")
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)

    # 复化辛普森规则的权重
    weights = np.ones(n + 1)
    weights[0] = weights[-1] = 1 / 3
    weights[1:-1:2] = 4 / 3
    weights[2:-1:2] = 2 / 3

    result = np.sum(weights * y) * h

    return result

def legendre_polynomial(n, x):
    """生成n阶勒让德多项式在x点的值"""
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        p0, p1 = 1, x
        for i in range(2, n + 1):
            p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
            p0, p1 = p1, p2
        return p1


def legendre_polynomial_derivative(n, x):
    """计算n阶勒让德多项式的导数在x点的值"""
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return n * (x * legendre_polynomial(n, x) - legendre_polynomial(n - 1, x)) / (x * x - 1)


def find_legendre_roots(n, tolerance=1e-12, max_iter=100):
    """使用牛顿法求解n阶勒让德多项式的根"""
    roots = []
    # 使用余弦函数的根作为初始猜测值
    for k in range(1, n + 1):
        x = math.cos(math.pi * (4 * k - 1) / (4 * n + 2))
        for _ in range(max_iter):
            p = legendre_polynomial(n, x)
            if abs(p) < tolerance:
                break
            dp = legendre_polynomial_derivative(n, x)
            x = x - p / dp
        roots.append(x)
    return roots


def gauss_legendre_weights(n):
    """计算Gauss-Legendre求积的权重"""
    roots = find_legendre_roots(n)
    weights = []
    for x in roots:
        dp = legendre_polynomial_derivative(n, x)
        w = 2 / ((1 - x * x) * dp * dp)
        weights.append(w)
    return roots, weights


def gauss_legendre(f, a, b, n):
    """Gauss-Legendre求积公式"""
    # 获取高斯点和权重
    nodes, weights = gauss_legendre_weights(n)

    # 区间变换
    result = 0
    for x, w in zip(nodes, weights):
        # 从[-1,1]变换到[a,b]
        transformed_x = 0.5 * (b - a) * x + 0.5 * (b + a)
        transformed_w = 0.5 * (b - a) * w
        result += transformed_w * f(transformed_x)

    return result


# 精确值
PI = 3.141592653589793
exact_value = PI / 2

# 1. 使用不同的复化求积公式
a, b = 0, 1
n_composite = 8  # 区间等分数

result_trapezoidal = composite_trapezoidal(f, a, b, n_composite)
result_simpson = composite_simpson(f, a, b, n_composite // 2)
result_cotes = composite_cotes(f, a, b, n_composite)

# 2. 使用Gauss-Legendre求积公式
n_values = [1, 3, 5]
results_gl = []
for n in n_values:
    result = gauss_legendre(f, a, b, n)
    results_gl.append(result)

# 3. 输出比较结果
print("数值积分结果比较 (保留小数点后10位):")
print("\n1. 复化求积公式结果:")
print(f"8阶复化梯形公式: {result_trapezoidal:.10f}, 误差: {abs(result_trapezoidal - exact_value):.10e}")
print(f"4阶复化Simpson公式: {result_simpson:.10f}, 误差: {abs(result_simpson - exact_value):.10e}")
print(f"2阶复化Cotes公式: {result_cotes:.10f}, 误差: {abs(result_cotes - exact_value):.10e}")

print("\n2. Gauss-Legendre求积结果:")
for n, result in zip(n_values, results_gl):
    print(f"n={n}: {result:.10f}, 误差: {abs(result - exact_value):.10e}")