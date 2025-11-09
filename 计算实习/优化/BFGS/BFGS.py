import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']  # 图中显示中文
plt.rcParams['axes.unicode_minus'] = False  # 图中正常显示负号
from scipy.optimize import minimize
def objective_function(x):
    A = np.diag([1,2,3])
    return 0.5*np.dot(x.T,np.dot(A,x))+1

def gradient(x):
    A = np.diag([1,2,3])
    return np.dot(A,x)
x0 = np.array([1,1,1])
result = minimize(objective_function,x0,method='BFGS',jac = gradient)

print('x:', result.x)
print(result.fun)