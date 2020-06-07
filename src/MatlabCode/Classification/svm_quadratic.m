function [alpha,intercept] = svm_quadratic(x,y,cost)
% 二分类SVM，注y属于{-1,1}，而不是{0,1}
% 需要增加的地方，若要使用时，为保持稀疏性，仅保留非0的alpha，即支持向量
% 软间隔支持向量机, 使用Matlab自带的二次规划函数quadprog求SVM的参数
% 利用此法无法找到确切的支持向量，设置一个阈值: 当1e-4 < alpha时, 对应一个支持向量
tic
m = size(x,1);
K = zeros(m); H = zeros(m);
for i = 1:m
    for j = 1:m
        xi = x(i,:); xj = x(j,:); 
        K(i,j) = kernel(xi,xj);
    end
end
% 计算H矩阵
for i = 1:m
    for j = 1:m
        H(i,j) = y(i) * y(j) * K(i,j);
    end
end
% 计算f,lower_bound, upper_bound, 约束条件
f = -ones(m,1);
Aeq = y'; beq = 0;
LB = zeros(m,1);
UB = cost * ones(m,1);

% 调二次优化包quadprog求解约束优化问题，即训练SVM，求出alpha的值
[alpha,~] = quadprog(H,f,[],[],Aeq,beq,LB,UB);

% 计算threshold
alpha_y = (alpha .* y)'; alpha_y_kernel = repmat(alpha_y,m,1) .* K;  
s_alpha_y_kernel = sum(alpha_y_kernel,2); 
MAX = max( s_alpha_y_kernel(y==-1 & ((alpha>1e-4)&(alpha<cost-1e-4)) )); 
MIN = min( s_alpha_y_kernel(y==+1 & ((alpha>1e-4)&(alpha<cost-1e-4)) )); 
intercept = -(MAX + MIN) / 2;

toc
end

%% 核函数
function kij = kernel(xi,xj)
kij = dot(xi,xj); % 线性核
% k_dot = exp((norm(xi-xj))^2 / gamma); % 高斯核
% k_dot = (dot(xi, xj) + a)^d; % 这是多项式核
end
