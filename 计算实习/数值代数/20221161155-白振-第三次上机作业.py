#20221161155-白振
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['STFangsong']  # 图中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 图中正常显示负号

###上机实验3
##第1题

def create_matrix(n):
    dim = (n - 1) ** 2
    A = np.zeros((dim, dim))

    # 每个分块矩阵的维数
    block_size = n - 1

    # A对角元为T_(n-1)
    for i in range(block_size):
        start_idx = i * block_size
        # 主对角线为4
        for j in range(block_size):
            A[start_idx + j, start_idx + j] = 4
        # 次对角线为1
        for j in range(block_size - 1):
            A[start_idx + j, start_idx + j + 1] = 1
            A[start_idx + j + 1, start_idx + j] = 1

    # A次对角线 -I_(n-1)
    for i in range(block_size - 1):
        start_idx = i * block_size
        next_idx = (i + 1) * block_size

        for j in range(block_size):
            A[start_idx + j, next_idx + j] = -1
            A[next_idx + j, start_idx + j] = -1

    return A


def create_b(n):
    dim = (n - 1) ** 2
    e = np.ones(dim)
    A = create_matrix(n)
    return A @ e


def jacobi(A, b, tol=1e-8, max_iter=1000):
    n = len(A)
    x = np.zeros(n)
    D = np.diag(A)
    R = A - np.diagflat(D)

    errors = []

    for i in range(max_iter):
        x_new = (b - R @ x) / D
        error = np.linalg.norm(b - A @ x_new)
        errors.append(error)

        if error < tol:
            return x_new, errors, i + 1
        x = x_new

    return x, errors, max_iter


def gauss_seidel(A, b, tol=1e-8, max_iter=1000):
    n = len(A)
    x = np.zeros(n)
    errors = []

    for iter_count in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]

        error = np.linalg.norm(b - A @ x_new)
        errors.append(error)

        if error < tol:
            return x_new, errors, iter_count + 1

        x = x_new

    return x, errors, max_iter


def sor(A, b, omega, tol=1e-8, max_iter=1000):
    n = len(A)
    x = np.zeros(n)
    errors = []

    for iter_count in range(max_iter):
        x_new = x.copy()

        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (1 - omega) * x[i] + (omega * (b[i] - s1 - s2)) / A[i, i]

        error = np.linalg.norm(b - A @ x_new)
        errors.append(error)

        if error < tol:
            return x_new, errors, iter_count + 1

        x = x_new

    return x, errors, max_iter
n = 10  # 矩阵维数 + 1
A = create_matrix(n)
b = create_b(n)
omega = 1.3


_, jacobi_errors, jacobi_steps = jacobi(A, b)
_, gs_errors, gs_steps = gauss_seidel(A, b)
_, sor_errors, sor_steps = sor(A, b, omega)

# 画图
plt.figure()
plt.semilogy(jacobi_errors, label=f'Jacobi ({jacobi_steps} 步)')
plt.semilogy(gs_errors, label=f'Gauss-Seidel ({gs_steps} 步)')
plt.semilogy(sor_errors, label=f'SOR ω=1.3 ({sor_steps} 步)')
plt.grid(True)
plt.xlabel('迭代步数')
plt.ylabel('迭代误差 (对数尺度)')
plt.legend()
plt.title('不同迭代方法的每步迭代误差比较')

# 画表 3.1
n = 4  # 9x9 矩阵
A = create_matrix(n)
b = create_b(n)
omegas = [0.2, 0.4, 0.6, 0.8, 1.0]

print("\nTable 3.1:不同松弛因子的收敛速度比较")
print(f"{'ω':<10} {'Steps':<10}{'n      '} {'ρ(G)':<10}{'Error':<15} ")
print("-" * 45)

for omega in omegas:
    _, errors, steps = sor(A, b, omega)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    G = np.linalg.inv(D / omega + L) @ ((1 - omega) / omega * D - U)
    rho = np.max(np.abs(np.linalg.eigvals(G)))

    print(f"{omega:<10.1f} {steps:<10d}{'4      '} {rho:<10.4f} {errors[-1]:<15.2e} ")


##第2题
import scipy.io
import scipy.sparse.linalg as sla
import time
from scipy.io import mmread

matrix_files = {
    'wiki-Vote': 'wiki-Vote.mtx',
    'web-Google': 'web-Google.mtx',
    'web-Stanford': 'web-Stanford.mtx'
}

matrices = {}
for name, file_path in matrix_files.items():
    matrix = mmread(file_path)
    matrices[name] = matrix


for name, A in matrices.items():

    if not scipy.sparse.issparse(A):
        A = scipy.sparse.csr_matrix(A)


    b = np.random.rand(A.shape[0])
omega = 1.3


_, jacobi_errors, jacobi_steps = jacobi(A, b)
_, gs_errors, gs_steps = gauss_seidel(A, b)
_, sor_errors, sor_steps = sor(A, b, omega)

# 画图
plt.figure()
plt.semilogy(jacobi_errors, label=f'Jacobi ({jacobi_steps} 步)')
plt.semilogy(gs_errors, label=f'Gauss-Seidel ({gs_steps} 步)')
plt.semilogy(sor_errors, label=f'SOR ω=1.3 ({sor_steps} 步)')
plt.grid(True)
plt.xlabel('迭代步数')
plt.ylabel('迭代误差 (对数尺度)')
plt.legend()
plt.title('不同迭代方法的每步迭代误差比较')

# 画表 3.1
n = 4  # 9x9 矩阵
A = create_matrix(n)
b = create_b(n)
omegas = [0.2, 0.4, 0.6, 0.8, 1.0]

print("\nTable 3.1:不同松弛因子的收敛速度比较")
print(f"{'ω':<10} {'Steps':<10}{'n      '} {'ρ(G)':<10}{'Error':<15} ")
print("-" * 45)

for omega in omegas:
    _, errors, steps = sor(A, b, omega)

    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    G = np.linalg.inv(D / omega + L) @ ((1 - omega) / omega * D - U)
    rho = np.max(np.abs(np.linalg.eigvals(G)))

    print(f"{omega:<10.1f} {steps:<10d}{'4      '} {rho:<10.4f} {errors[-1]:<15.2e} ")


##第二题
import numpy as np
import time
from pyamg.relaxation.relaxation import jacobi, gauss_seidel, sor
from scipy.io import mmread
from scipy.sparse import eye, diags
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.sans-serif'] = ['STFangsong']  # 图中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 图中正常显示负号


def read_mtx_file(file_path):
    sparse_matrix = mmread(file_path)
    row_sums = sparse_matrix.sum(axis=1).A.flatten()
    row_sums[row_sums == 0] = 1
    row_diag = diags(1.0 / row_sums)
    return row_diag @ sparse_matrix


class IterativeSolver:
    def __init__(self, M, b, x0, tol=1e-8, max_iter=1000):
        self.M = M
        self.b = b
        self.x0 = x0.copy()
        self.tol = tol
        self.max_iter = max_iter
        self.residuals = []
        self.times = []
        self.iterations = []

    def solve(self, method, method_name, **kwargs):
        x = self.x0.copy()
        start_time = time.time()

        for i in tqdm(range(self.max_iter), desc=f"{method_name}计算中"):
            method(self.M, x, self.b, **kwargs)
            residual = np.linalg.norm(self.M @ x - self.b)
            self.residuals.append(residual)
            self.times.append(time.time() - start_time)
            self.iterations.append(i + 1)

            if residual < self.tol:
                break

        return {
            "迭代次数": i + 1,
            "计算时间": time.time() - start_time,
            "最终残差": residual,
            "解": x
        }


def plot_convergence(solvers, names):
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 1, 1)
    for solver, name in zip(solvers, names):
        plt.semilogy(solver.iterations, solver.residuals, label=name)
    plt.grid(True)
    plt.xlabel('迭代次数')
    plt.ylabel('残差 (对数尺度)')
    plt.title('迭代方法收敛性对比')
    plt.legend()

    plt.subplot(2, 1, 2)
    for solver, name in zip(solvers, names):
        plt.plot(solver.times, solver.residuals, label=name)
    plt.grid(True)
    plt.xlabel('计算时间 (秒)')
    plt.ylabel('残差')
    plt.title('迭代方法性能对比')
    plt.legend()

    plt.tight_layout()
    plt.show()


# 主程序
matrix_files = {
    'Wiki投票': 'wiki-Vote.mtx',
    'Google网页': 'web-Google.mtx',
    'Stanford网页': 'web-Stanford.mtx'
}

for name, file_name in matrix_files.items():
    print(f"\n处理矩阵: {name}")
    A = read_mtx_file(file_name)
    print(f"矩阵维度: {A.shape}")

    alpha = 0.85
    n = A.shape[0]
    I = eye(n, format='csr')
    b = np.ones(n) / n
    M = I - alpha * A

    x0 = np.zeros(n)
    tol = 1e-8
    max_iter = 1000

    solvers = []
    solver_names = []

    jacobi_solver = IterativeSolver(M, b, x0, tol=tol, max_iter=max_iter)
    jacobi_result = jacobi_solver.solve(jacobi, "Jacobi方法")
    solvers.append(jacobi_solver)
    solver_names.append("Jacobi方法")

    gs_solver = IterativeSolver(M, b, x0, tol=tol, max_iter=max_iter)
    gs_result = gs_solver.solve(gauss_seidel, "Gauss-Seidel方法", sweep='forward')
    solvers.append(gs_solver)
    solver_names.append("Gauss-Seidel方法")

    sor_solver = IterativeSolver(M, b, x0, tol=tol, max_iter=max_iter)
    sor_result = sor_solver.solve(sor, "SOR方法", omega=1.1, sweep='forward')
    solvers.append(sor_solver)
    solver_names.append("SOR方法")

    plot_convergence(solvers, solver_names)

    for solver, name in zip(solvers, solver_names):
        print(f"\n{name}:")
        print(f"  迭代次数: {len(solver.iterations)}")
        print(f"  计算时间: {solver.times[-1]:.4f} 秒")
        print(f"  最终残差: {solver.residuals[-1]:.4e}")