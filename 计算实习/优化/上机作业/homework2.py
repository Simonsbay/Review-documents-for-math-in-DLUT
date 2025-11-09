import numpy as np
from imageio.v2 import sizes


def fun(x):
    n = 155
    G = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                G[i, j] = 2 * (i+1)
            else:
                G[i, j] = min(i+1, j+1)
    G = np.array(G)
    one_vector = np.ones(n)
    c = 0.5 * np.dot(G, one_vector)
    f = 0.5 * x.T @ G @ x + c.T @ x
    g = np.dot(G, x) + c  # 梯度计算
    return f, g, G


def steep(x0, eps):
    _, gk, G = fun(x0)
    res = np.linalg.norm(gk)
    iter = 0
    xk = x0  # 初始化xk为x0
    while res > eps and iter < 10000:
        iter += 1
        dk = -gk  # 下降方向
        alphak = np.dot(gk.T, gk) / np.dot(gk.T, np.dot(G,gk))  # 步长
        xk = xk + alphak * dk  # 更新xk
        _, gk, _ = fun(xk)
        res = np.linalg.norm(gk)  # 计算新的残差
        print('{}th iteration, the residual is {}'.format(iter, res))
    print('steep最优解为:',xk)
    return xk
#steep(np.zeros(155), 1e-3)

def newton(x0, eps):
    _, gk, G = fun(x0)
    res = np.linalg.norm(gk)
    xk = x0
    iter = 0
    print('{}th iteration,the residual is --{}'.format(iter, res))
    while res > eps and iter < 1000:
        iter += 1
        alphak = 1
        dk= np.linalg.solve(G, -gk)
        xk = xk + alphak * dk
        _, gk, _ = fun(xk)
        res = np.linalg.norm(gk)
        print('{}th iteration,the residual is --{}'.format(iter, res))
    print('newton最优解为:', xk)
    return xk
#newton(np.zeros(155),1e-4)


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
        alphak = np.dot(-gk.T, dk) / np.dot(np.dot(dk.T, G), dk)  #
        xk += np.dot(alphak, dk)
        deltak = np.dot(alphak, dk)
        gk0 = gk
        _, gk, _ = fun(xk)
        yk = gk - gk0
        Hk = Hk - np.outer(Hk @ yk, yk @ Hk) / (yk.T @ Hk @ yk) + np.outer(deltak, deltak) / (deltak.T @ yk)
        res = np.linalg.norm(gk)
        print('{}th iteration,the residual is --{}'.format(iter, res))
    print('dfp最优解为:', xk)
    return xk
dfp(np.zeros(155), 1e-4)


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
        ak = np.dot(-gk.T, dk) / np.dot(np.dot(dk.T, G), dk)
        xk = xk + np.dot(ak, dk)
        deltak = np.dot(ak, dk)
        gk0 = gk
        _, gk, _ = fun(xk)
        yk = gk - gk0
        Bk = Bk + np.outer(yk,yk)/np.dot(yk.T,deltak) - np.outer(np.dot(Bk,deltak),np.dot(Bk,deltak))/np.dot(deltak.T,np.dot(Bk,deltak))
        res = np.linalg.norm(gk)
        print('{}th iteration,the residual is --{}'.format(iter, res))
    print('bfgs最优解为:', xk)
    return xk
#bfgs(np.zeros(155), 1e-4)

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
        alphak = np.dot(-gk.T, dk) / np.dot(np.dot(dk.T, G), dk)  #
        xk = xk + np.dot(alphak,dk)
        _, gk, _ = fun(xk)
        res0 = res
        res = np.linalg.norm(gk)
        print('{}th iteration,the residual is --{}'.format(iter, res))
    print('fr最优解为:', xk)
    return xk
#fr(np.zeros(155),1e-4)
