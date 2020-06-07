function [alpha,intercept] = svm_quadratic(x,y,cost)
% ������SVM��עy����{-1,1}��������{0,1}
% ��Ҫ���ӵĵط�����Ҫʹ��ʱ��Ϊ����ϡ���ԣ���������0��alpha����֧������
% �����֧��������, ʹ��Matlab�Դ��Ķ��ι滮����quadprog��SVM�Ĳ���
% ���ô˷��޷��ҵ�ȷ�е�֧������������һ����ֵ: ��1e-4 < alphaʱ, ��Ӧһ��֧������
tic
m = size(x,1);
K = zeros(m); H = zeros(m);
for i = 1:m
    for j = 1:m
        xi = x(i,:); xj = x(j,:); 
        K(i,j) = kernel(xi,xj);
    end
end
% ����H����
for i = 1:m
    for j = 1:m
        H(i,j) = y(i) * y(j) * K(i,j);
    end
end
% ����f,lower_bound, upper_bound, Լ������
f = -ones(m,1);
Aeq = y'; beq = 0;
LB = zeros(m,1);
UB = cost * ones(m,1);

% �������Ż���quadprog���Լ���Ż����⣬��ѵ��SVM�����alpha��ֵ
[alpha,~] = quadprog(H,f,[],[],Aeq,beq,LB,UB);

% ����threshold
alpha_y = (alpha .* y)'; alpha_y_kernel = repmat(alpha_y,m,1) .* K;  
s_alpha_y_kernel = sum(alpha_y_kernel,2); 
MAX = max( s_alpha_y_kernel(y==-1 & ((alpha>1e-4)&(alpha<cost-1e-4)) )); 
MIN = min( s_alpha_y_kernel(y==+1 & ((alpha>1e-4)&(alpha<cost-1e-4)) )); 
intercept = -(MAX + MIN) / 2;

toc
end

%% �˺���
function kij = kernel(xi,xj)
kij = dot(xi,xj); % ���Ժ�
% k_dot = exp((norm(xi-xj))^2 / gamma); % ��˹��
% k_dot = (dot(xi, xj) + a)^d; % ���Ƕ���ʽ��
end