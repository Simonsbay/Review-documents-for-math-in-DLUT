function [f_min, x_star] = APG(x0, A, b, r, n, eps)
% 加速的临近梯度法（APG）函数，用于求解特定优化问题
% 输入参数：
%   x0：初始解向量
%   A：系数矩阵
%   b：右侧向量
%   r：与正则化项相关的参数
%   n：向量的维度
%   eps：收敛阈值

% 计算矩阵A'*A的特征值，取其中最大值作为L用于确定步长
lambda = eig(A'*A); 
L = max(lambda); 

% 计算步长tk，根据矩阵A'*A的最大特征值确定
tk = 1/L; 

% 将初始解向量x0赋值给xk，作为当前迭代的x值的初始值
xk = x0; 

% 初始化yk，按照特定公式进行赋值，这是加速临近梯度法中的一步操作
yk = xk - 1/2*(xk - x0);

% 计算梯度gk，根据公式A'*(A*yk - b)计算，其中yk是当前的中间变量
gk = A'*(A*yk - b); 

% 根据临近梯度法的更新公式更新xk，这里先基于梯度gk进行更新
xk = yk - tk*gk;

% 对xk的每个分量进行处理，根据一定规则进行收缩操作，可能与正则化项相关
for i = 1 : n
    xk(i) = sign(xk(i))*max(abs(xk(i)) - tk*r, 0);
end

% 计算初始残差，即当前xk与初始解x0的差值的范数
res = norm(xk - x0);
% 迭代次数计数器，初始化为0
iter = 0; 

% 打印初始迭代时的残差信息
fprintf('At %d-th iteration,the residual is -------------%d\n', iter, res);

% 迭代循环条件：当残差大于收敛阈值eps且迭代次数小于10000时继续迭代
while res > eps && iter < 10000
    iter = iter + 1; % 迭代次数加1

    x0 = xk; % 保存上一次迭代的xk值，用于后续计算残差

    % 根据加速临近梯度法的公式更新yk，其中涉及到当前迭代次数iter
    yk = xk + (iter - 2)/(iter + 1)*(xk - x0);

    % 重新计算梯度gk，因为yk已经更新
    gk = A'*(A*yk - b);

    % 根据临近梯度法的更新公式再次更新xk，基于更新后的yk和gk
    xk = yk - tk*gk;

    % 再次对xk的每个分量进行收缩操作，与之前的规则相同
    for i = 1 : n
        xk(i) = sign(xk(i))*max(abs(xk(i)) - tk*r, 0);
    end

    % 重新计算残差，即当前xk与上一次迭代保存的x0的差值的范数
    res = norm(xk - x0);

    % 打印当前迭代时的残差信息
    fprintf('At %d-th iteration,the residual is -------------%d\n', iter, res);
end

% 迭代结束后，将最终的xk值作为最优解x_star
x_star = xk;

% 计算最优解x_star对应的目标函数值f_min，目标函数包含数据拟合项和正则化项
f_min = 1/2*norm(A*x_star - b)^2 + r*norm(x_star, 1);

end