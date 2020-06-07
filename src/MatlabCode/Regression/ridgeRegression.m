% ridge regression
% 待完善之处：增加系数的t检验
function [theta,ASC,R_square] = ridgeRegression(x,y,lamda,algorithm)
if(nargin < 3); error('输入的参数不能少于3个');end
if(nargin == 3); 
    [theta,ASC,R_square] = moment_Method(x,y,lamda); return 
end
if(strcmp(algorithm,'moment') == 1)
   [theta,ASC,R_square] = moment_Method(x,y,lamda);
elseif(strcmp(algorithm,'grad') == 1)
   [theta,ASC,R_square] = grad_dec(x,y,lamda);
end
end

function [theta,ASC,R_square] = moment_Method(x,y,lamda)
% 使用矩法
[m,p] = size(x);
X = [ones(m,1) x];
theta = (y'*X) / (X'*X + lamda*eye(p+1));
R_square = var(X*theta') / var(y);
ASC = theta(1); theta = theta(2:end);
end

function [theta,ASC,R_square] = grad_dec(x,y,lamda)
% 使用梯度下降法
[m,p] = size(x);
X = [ones(m,1) x];
theta = zeros(1,p+1); v_theta = zeros(1,p+1);
epoch = 1e4;
momentum = 0.9; lr = 0.01;
for i = 1:epoch
    d_theta = theta*X'*X - y'*X + lamda*theta;
    v_theta = momentum * v_theta - lr * d_theta;
    theta = theta + v_theta;
end
R_square = var(X*theta') / var(y);
ASC = theta(1); theta = theta(2:end);
end




