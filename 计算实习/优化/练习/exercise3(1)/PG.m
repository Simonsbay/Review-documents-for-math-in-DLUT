function [f_min,x_star] = PG(x0,A,b,r,n,eps)
%临近梯度法

lambda = eig(A'*A); L = max(lambda); %%李普希茨常数

tk = 1/L; %%步长
xk = x0; %%赋值
gk = A'*(A*xk - b); %%梯度
yk = xk - tk*gk;

for i = 1:n
    xk(i) = sign(yk(i))*max(abs(yk(i))-tk*r, 0);
end

res = norm(xk-x0);
iter = 0; %%计数
fprintf('At %d-th iteration,the residual is -------------%d\n',iter,res);

while res > eps && iter < 10000
    iter = iter + 1;

    x0 = xk; gk = A'*(A*xk - b);
    yk = xk - tk*gk;
    
    for i = 1:n
        xk(i) = sign(yk(i))*max(abs(yk(i))-tk*r, 0);
    end

    gk = A'*(A*xk - b); res = norm(xk - x0);
    fprintf('At %d-th iteration,the residual is -------------%d\n',iter,res);
end

x_star = xk;
f_min = 1/2*norm(A*x_star-b)^2 + r*norm(x_star,1);

end

