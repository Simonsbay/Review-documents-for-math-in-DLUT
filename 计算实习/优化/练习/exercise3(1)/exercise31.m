
%%随机生成A，b,r
n = 256; m = 128;
A = randn(m,n);
u = sprandn(n,1,0.1);
b = A*u; 
r = 10;
eps = 1e-5;
x = zeros([n,1]);
[f_min1,x1] = PG(x,A,b,r,n,eps);
disp('最优解 x: ');
disp(x1);
disp('最小目标函数值: ');
disp(f_min1);

[f_min2,x2] = APG(x,A,b,r,n,eps);
disp('最优解 x: ');
disp(x1);
disp('最小目标函数值: ');
disp(f_min1);

[f_min3,x3] = ADMM(x,A,b,r,n,eps);
disp('最优解 x: ');
disp(x3);
disp('最小目标函数值: ');
disp(f_min1);
