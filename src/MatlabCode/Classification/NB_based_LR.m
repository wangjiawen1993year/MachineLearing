% Naive Bayesian + LR
% rng(1);x = rand(2100,100); y = sum(x,2)>50;  x_train = x(1:2000,:); 
% x_test = x(2001:end,:); y_train = y(1:2000); y_test = y(2001:end);
function [para_NB,w,b,error_rate,probs] = NB_based_LR(x_train,y_train,x_test,y_test)
% 必须输入训练数据，测试数据可以不输入
if(nargin < 2);error('输入参数不能少于2个！');end
% 计算参数（基于MLE的思想）
para_NB.miu0 = mean(x_train(y_train==0,:),1); para_NB.sigma0 = std(x_train(y_train==0,:),1);
para_NB.miu1 = mean(x_train(y_train==1,:),1); para_NB.sigma1 = std(x_train(y_train==1,:),1);
para_NB.prior_of_positive = sum(y_train==1) / size(y_train,1);
% 计算训练集的error rate
    % 调用下面的子函数，计算后验概率p(y|x)与error_rate
sum_log = log_pxgiveny(x_train,para_NB,y_train);
[sum_log,mu,sigma] = zscore(sum_log);
% [w,b,error_rate.TrainingSet] = MLP(zscore_x,y_train,[2,1]);
% w = w{1}; b = b{1};
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
w = 0.1*randn(size(sum_log,2),size(y_train,2)); b =0;
vw = zeros(size(w)); vb = 0;
% 正则化项系数
lamda = 0;
num_of_epoch = 1e4;
loss = zeros(num_of_epoch,1);
for i =1:num_of_epoch
    y_pred = sigm(sum_log,w,b);
    dw = (sum_log' * (y_train.* (1-y_pred) - (1-y_train) .* y_pred) - lamda * w) / size(sum_log,1);
    db = mean(y_train.* (1-y_pred) - (1-y_train) .* y_pred - lamda, 1);
    vw = 0.9*vw + dw; vb = 0.9*vb + 0.1*db;
    w = w + vw;
    b = b + vb;
    % 记录loss
    y_pred1 = sigm(sum_log,w,b);
    if(sum(y_pred1==1)>0) % 防止出现NaN的“平滑”操作
        y_pred1(y_pred1==1) = y_pred1(y_pred1==1) - eps;
    end
    loss(i,1) = mean(-y_train.*log(y_pred1) - (1-y_train).* log(1-y_pred1),1);
end
% 计算训练结果

% 计算测试结果
if(nargin == 3)    
    sum_log = log_pxgiveny(x_test,para_NB,ones(size(x_test),1));
    sum_log = (sum_log - repmat(mu,ones(size(x_test),1))) ./ repmat(sigma,ones(size(x_test),1));
    probs = 1 ./ (1 + exp(-sum_log * w - b));
elseif(nargin == 4)
    sum_log = log_pxgiveny(x_test,para_NB,ones(size(y_test)));
    sum_log = (sum_log - repmat(mu,size(y_test))) ./ repmat(sigma,size(y_test));
    probs = 1 ./ (1 + exp(-sum_log * w - b));
    y_pred = probs > 0.5;
    error_rate.TestSet = sum(xor(y_pred,y_test)) / size(y_test,1);
end
sum_log = log_pxgiveny(x_train,para_NB,ones(size(y_train)));
sum_log = (sum_log - repmat(mu,size(y_train))) ./ repmat(sigma,size(y_train));
probs = 1 ./ (1 + exp(-sum_log * w - b));
y_pred = probs > 0.5;
error_rate.TrainSet = sum(xor(y_pred,y_train)) / size(y_train,1);
display(error_rate);

plot(loss)
display(loss(end))
end


function sum_log = log_pxgiveny(x,para_NB,y)
kao = 1e-10; % p平滑系数
prior_OrNot = 0; 
prior = para_NB.prior_of_positive;
miu0 = para_NB.miu0; sigma0 = para_NB.sigma0; 
miu1 = para_NB.miu1; sigma1 = para_NB.sigma1;
fai0 = log(normpdf(x,repmat(miu0,size(x,1),1),repmat(sigma0,size(x,1),1)) + kao);
avg = diag(1-y) * repmat(miu0,size(x,1),1) + diag(y) * repmat(miu1,size(x,1),1); 
stddev = diag(1-y) * repmat(sigma0,size(x,1),1) + diag(y) * repmat(sigma1,size(x,1),1);
fai_y = log(normpdf(x,avg,stddev) +  kao);
sum_log = [sum(fai0,2)+ prior_OrNot * log(1-prior),...
           sum(fai_y,2)+ prior_OrNot * (y *log(prior) + (1-y) *log(1-prior))];
end





