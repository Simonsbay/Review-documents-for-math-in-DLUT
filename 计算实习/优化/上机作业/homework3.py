import numpy as np
#函数1
def fun(x):
    f = (10000*x[0]*x[1]-1)**2+(np.exp(-x[0])+np.exp(-x[1])-1.0001)**2
    g = np.array([2*(10000*x[0]*x[1]-1)*(10000*x[1])+2*(np.exp(-x[0])+np.exp(-x[1])-1.0001)*(-np.exp(-x[0])), 2*(10000*x[0]*x[1]-1)*(10000*x[0])+2*(np.exp(-x[0])+np.exp(-x[1])-1.0001)*(-np.exp(-x[1]))])
    G = np.array([
        [
            2 * (10000 ** 2) * x[1] ** 2 + 2 * np.exp(-2 * x[0]) + 2 * (np.exp(-x[0]) + np.exp(-x[1]) - 1.0001) * np.exp(-x[0]),
            2 * (10000 ** 2) * x[0] * x[1] + 2 * np.exp(-x[0] - x[1])
        ],
        [
            2 * (10000 ** 2) * x[0] * x[1] + 2 * np.exp(-x[0] - x[1]),
            2 * (10000 ** 2) * x[0] ** 2 + 2 * np.exp(-2 * x[1]) + 2 * (np.exp(-x[0]) + np.exp(-x[1]) - 1.0001) * np.exp(-x[1])
        ]
    ])
    return f, g, G
def f(x):
    return (10000*x[0]*x[1]-1)**2+(np.exp(-x[0])+np.exp(-x[1])-1.0001)**2

def steep(x0, eps):
    _, gk, G = fun(x0)
    res = np.linalg.norm(gk)
    iter = 0
    xk = x0
    while res > eps and iter <10000:
        iter += 1
        dk = -gk
        alpha = 1.0
        beta = 0.5
        c = 0.5
        while f(xk + alpha * dk) > f(xk) + c * alpha * np.dot(gk.T, dk):
            alpha *= beta
        xk = xk + alpha * dk
        _, gk, _ = fun(xk)
        res = np.linalg.norm(gk)
        print('{}th iteration, the residual is {}'.format(iter, res))
    print('steep 最优解为:', xk)
    return xk
#steep([0,1], 1e-4)





def newton(x0, eps):
    _, gk, G = fun(x0)
    res = np.linalg.norm(gk)
    xk = x0
    iter = 0
    print('{}th iteration, the residual is {}'.format(iter, res))

    while res > eps and iter < 10000:
        iter += 1
        dk = np.linalg.solve(G, -gk)
        x0 = xk
        ak = 1
        f0, _, _ = fun(x0)
        xk = x0 + ak * dk
        f1, _, _ = fun(xk)
        while f1 > f0 + 0.1 * ak * (gk @ dk):
            ak = ak / 2
            xk = x0 + ak * dk
            f1, _, _ = fun(xk)
        _, gk, G = fun(xk)
        res = np.linalg.norm(gk)
        print('{}th iteration, the residual is {}'.format(iter, res))
    print('Newton 最优解为:', xk)
    return xk
#newton([0,1],1e-4)



def dfp(x0, eps):
    n = len(x0)
    Hk = np.eye(n)
    _, gk, G = fun(x0)
    res = np.linalg.norm(gk)
    xk = x0
    iter = 0
    print('{}th iteration,the residual is --{}'.format(iter, res))

    while res > eps and iter < 10000:
        iter += 1
        dk = np.dot(-Hk, gk)
        alpha = 1.0
        beta = 0.5
        c = 0.5
        while f(xk + alpha * dk) > f(xk) + c * alpha * np.dot(gk, dk):
            alpha *= beta
        xk += alpha * dk
        deltak = np.dot(alpha, dk)
        gk0 = gk
        _, gk, _ = fun(xk)
        yk = gk - gk0
        Hk = Hk - np.outer(Hk @ yk, yk @ Hk) / (yk.T @ Hk @ yk) + np.outer(deltak, deltak) / (deltak.T @ yk)
        res = np.linalg.norm(gk)
        print('{}th iteration,the residual is --{}'.format(iter, res))
    print('dfp最优解为:', xk)
    return xk
#dfp([0,1], 1e-4)


def bfgs(x0, eps):
    n = len(x0)
    Bk = np.eye(n)
    _, gk, G = fun(x0)
    res = np.linalg.norm(gk)
    xk = x0
    iter = 0
    print('{}th iteration,the residual is --{}'.format(iter, res))
    while res > eps and iter < 10000:
        iter += 1
        dk = np.linalg.solve(Bk, -gk)
        alpha = 1.0
        beta = 0.5
        c = 0.5
        while f(xk + alpha * dk) > f(xk) + c * alpha * np.dot(gk, dk):
            alpha *= beta
        xk = xk + alpha * dk
        deltak = np.dot(alpha, dk)
        gk0 = gk
        _, gk, _ = fun(xk)
        yk = gk - gk0
        Bk = Bk + np.outer(yk,yk)/np.dot(yk.T,deltak) - np.outer(np.dot(Bk,deltak),np.dot(Bk,deltak))/np.dot(deltak.T,np.dot(Bk,deltak))
        res = np.linalg.norm(gk)
        print('{}th iteration,the residual is --{}'.format(iter, res))
    print('bfgs最优解为:', xk)
    return xk
#bfgs([0,1], 1e-4)


def fr(x0, eps):
    _, gk, G = fun(x0)
    res = np.linalg.norm(gk)
    xk = x0
    iter = 0
    print('{}th iteration,the residual is {}'.format(iter, res))

    while res > eps and iter < 10000:
        iter += 1
        if iter == 1:
            dk = -gk  # direction
        else:
            beta = np.dot(res, res) / np.dot(res0, res0)
            dk = -gk + beta * dk
        alpha = 1.0
        beta = 0.5
        c = 0.5
        while f(xk + alpha * dk) > f(xk) + c * alpha * np.dot(gk, dk):
            alpha *= beta
        xk += alpha * dk
        _, gk, _ = fun(xk)
        res0 = res
        res = np.linalg.norm(gk)
        print('{}th iteration,the residual is --{}'.format(iter, res))
    print('FR 最优解为:', xk)
    return xk
#fr([0,1], 1e-4)


def bb(x0, eps):
    n = len(x0)
    xk = x0
    f, gk, _ = fun(xk)
    res = np.linalg.norm(gk)
    iter = 0
    print('{}th iteration, the residual is --{}'.format(iter, res))
    # 初始步长
    alpha = 1.0
    while res > eps and iter < 10000:
        iter += 1
        # BB方法的两种步长计算方式
        if iter > 1:
            sk = xk - xk_old
            yk = gk - gk_old

            # BB1步长
           # alpha = np.abs((sk.T @ sk) / (sk.T @ yk))
            # 或者使用BB2步长
            alpha = np.abs((sk.T @ yk) / (yk.T @ yk))
        # 保存当前点和梯度
        xk_old = xk.copy()
        gk_old = gk.copy()
        # 沿负梯度方向更新
        xk = xk - alpha * gk
        # 计算新的函数值和梯度
        f, gk, _ = fun(xk)
        res = np.linalg.norm(gk)
        print('{}th iteration, the residual is --{}'.format(iter, res))
    print('BB 最优解为:', xk)
    return xk
#bb([0,1], 1e-4)