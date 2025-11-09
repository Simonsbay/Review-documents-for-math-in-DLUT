#20221161155-白振
###上机实验题1
##第1题
from decimal import Decimal, getcontext

def Sn_ascending(n):
    S = 0
    for k in range(2, n + 1):
        S += 10**7 / (k**2 - 1)
    return S

S_ascending = Sn_ascending(10**6)
print("S_10^6 (从小到大):", S_ascending)

def Sn_descending(n):
    S = 0
    for k in range(n, 1, -1):
        S += 10**7 / (k**2 - 1)
    return S

S_descending = Sn_descending(10**6)
print("S_10^6 (从大到小):", S_descending)


getcontext().prec = 28  # 设置精度

def significant_digits(number):
    return int(Decimal(str(number)).normalize().as_tuple().exponent * -1)

# 计算有效位数
sd_ascending = significant_digits(S_ascending)
sd_descending = significant_digits(S_descending)

print("有效位数 (从小到大):", sd_ascending)
print("有效位数 (从大到小):", sd_descending)
import numpy as np


##第2题
##Schur分解
def householder_vector(x):
    n = len(x)
    v = np.copy(x)
    sigma = np.sign(x[0]) * np.linalg.norm(x)
    v[0] = v[0] + sigma
    v = v / np.linalg.norm(v)
    return v

def hessenberg(A):
    n = A.shape[0]
    H = np.copy(A)
    Q = np.eye(n)

    for k in range(n - 2):
        x = H[k + 1:, k]
        if np.linalg.norm(x) > 1e-15:
            v = householder_vector(x)

            P = np.eye(n)
            P[k + 1:, k + 1:] -= 2.0 * np.outer(v, v)

            H = P @ H @ P
            Q = Q @ P

    return H, Q


def qr_iteration(H, Q, max_iter=100, tol=1e-10):
    n = H.shape[0]
    T = np.copy(H)
    U = np.copy(Q)

    for iter in range(max_iter):
        mu = T[-1, -1]  # 位移
        Q_k, R = np.linalg.qr(T - mu * np.eye(n))
        T = R @ Q_k + mu * np.eye(n)
        U = U @ Q_k

        # 检查收敛性
        if np.all(np.abs(np.tril(T, -1)) < tol):
            break

    return T, U


def schur(A, max_iter=100, tol=1e-10):
    H, Q = hessenberg(A)

    T, U = qr_iteration(H, Q, max_iter, tol)

    return T, U


np.random.seed(42)
A = np.random.rand(4, 4)
# A = np.array([[5, 4, 2], [1, 3, 1], [2, 1, 4]], dtype=float)
# 计算Schur分解
T, U = schur(A)

# 验证结果
print("原始矩阵 A:")
print(A)
print("\nSchur三角形式 T:")
print(T)
print("\n酉矩阵 U:")
print(U)
print("\n验证 A = U @ T @ U.T:")
print(U @ T @ U.T)
print("\n误差:")
print(np.max(np.abs(A - U @ T @ U.T)))
print("验证：A = Q * T * Q^H")
print(np.allclose(A, U @ T @ U.T.conj(), atol=1e-5))

##奇异值分解
import numpy as np


def power_iteration(A, num_iterations=100, tolerance=1e-10):
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iterations):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)

        if np.allclose(v, v_new, rtol=tolerance):
            break
        v = v_new

    return v


def svd(A):
    m, n = A.shape
    k = min(m, n)

    U = np.zeros((m, k))
    S = np.zeros(k)
    VT = np.zeros((k, n))

    AT = A.T
    X = A.copy()

    for i in range(k):
        v = power_iteration(AT @ A)
        u = A @ v
        sigma = np.linalg.norm(u)
        u = u / sigma

        U[:, i] = u
        S[i] = sigma
        VT[i, :] = v

        outer_product = np.outer(u, v)
        X = X - sigma * outer_product
        A = X
        AT = A.T
    return U, S, VT


# 测试矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

U, S, VT = svd(A)

# 验证结果
reconstructed = U @ np.diag(S) @ VT
print("原矩阵:\n", A)
print("\n重构矩阵:\n", reconstructed)