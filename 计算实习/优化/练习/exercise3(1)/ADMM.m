function [f_min, x_star] = ADMM(x0, A, b, r, n, eps)
% 交替方向乘子法（ADMM）函数，用于求解特定优化问题
% 输入参数：
%   x0：初始解向量
%   A：系数矩阵
%   b：右侧向量
%   r：与正则化项相关的参数
%   n：向量的维度
%   eps：收敛阈值

% 初始化迭代变量
xk = x0; % 当前迭代的x值，初始化为给定的初始解x0
yk = x0; % 对偶变量y，初始化为给定的初始解x0
zk = x0; % 辅助变量z，初始化为给定的初始解x0

res = norm(xk - x0); % 计算初始残差，即当前x与初始解x0的差值的范数

tao = 1.618; % ADMM算法中的参数tao，这里取值为黄金分割比
rho = 0.1; % ADMM算法中的惩罚参数rho

iter = 0; % 迭代次数计数器，初始化为0

fprintf('At %d-th iteration,the residual is -------------%d\n', iter, res); 
% 打印初始迭代时的残差信息

while iter == 0 || res > eps
    % 迭代循环条件：当为第一次迭代（iter == 0）或者残差大于收敛阈值eps时继续迭代

    iter = iter + 1; % 迭代次数加1

    x0 = xk; % 保存上一次迭代的x值，用于后续计算残差

    % 更新x的迭代公式，根据ADMM算法的标准更新步骤
    xk = (A' * A + rho * eye(n))^(-1) * (A' * b + rho * zk - yk);

    s = xk + yk / rho;  
    % 中间计算步骤，用于后续更新z

    for i = 1 : n
        % 循环更新z的每个分量
        zk(i) = sign(s(i)) * max(abs(s(i)) - r / rho, 0);
        % 根据ADMM算法的收缩算子更新z的第i个分量
    end

    yk = yk + tao * rho * (xk - zk);
    % 更新对偶变量y，根据ADMM算法的对偶变量更新公式

    res = norm(xk - x0); 
    % 重新计算残差，即当前x与上一次迭代保存的x0的差值的范数

    fprintf('At %d-th iteration,the residual is -------------%d\n', iter, res);
    % 打印当前迭代时的残差信息
end

x_star = xk; 
% 迭代结束后，将最终的x值作为最优解x_star

f_min = 1 / 2 * norm(A * xk - b) + r * norm(xk, 1);
% 计算最优解对应的目标函数值f_min，目标函数包含数据拟合项和正则化项
end
