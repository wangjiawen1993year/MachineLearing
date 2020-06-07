function [w,b,probs_of_positive,error_rate_of_TestSet] = logistic_regression(x_train,y_train,x_test,y_test)
% 用BP算法的思想训练感知器（这里是logistic regression）
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
w = 0.1 * randn(size(x_train,2),size(y_train,2)); b = 0; 
vw = zeros(size(w)); vb = 0;
num_of_epoch = 1e4;
loss = zeros(num_of_epoch,1);

% 小批量梯度下降
batch_size = size(x_train,1); % 默认使用全梯度
num_of_sample = size(x_train,1);
remainder = mod(num_of_sample, batch_size);
start = 1 : batch_size : (num_of_sample-remainder);
final = batch_size : batch_size : (num_of_sample-remainder);
final(end) = final(end) + remainder;
batch_index = [start;final]';
for i =1:num_of_epoch
    j = mod(i,size(batch_index,1)) + 1 ; %选batch
    x_batch = x_train(batch_index(j,1):batch_index(j,2),:); y_batch = y_train(batch_index(j,1):batch_index(j,2),:);
    y_pred = sigm(x_batch,w,b);
    % dw = x' * ((y./y_pred - (1-y)./(1-y_pred)) .* y_pred .* (1-y_pred)) / size(x,1); % 这是不稳定的，会出现NaN
    % db = mean((y./y_pred - (1-y)./(1-y_pred)) .* y_pred .* (1-y_pred), 1);           % 因此，不建议使用BP训练LR
    dw = x_batch' * (y_batch .* (1-y_pred) - (1-y_batch) .* y_pred) / size(x_batch,1);
    db = mean(y_batch .* (1-y_pred) - (1-y_batch) .* y_pred, 1);
    vw = 0.9*vw + 0.1*dw; vb = 0.9*vb + 0.1*db;
    w = w + vw;
    b = b + vb;
    y_pred = sigm(x_train,w,b);
    loss(i) = mean(-y_train.*log(y_pred) - (1-y_train).* log(1-y_pred),1);    
end
plot(loss)
[probs_of_positive, error_rate_of_TestSet] = predict(w,b,x_test,y_test);
display(error_rate_of_TestSet)
final_loss = min(loss);
display(final_loss)
end

function [probs_of_positive, error_rate_of_TestSet] = predict(w,b,x_test,y_test)
sigm = @(x,w,b) 1 ./ (1 + exp(-x*w - repmat(b,size(x,1),1)));
threshold = 0.5;
probs_of_positive = sigm(x_test,w,b);
error_rate_of_TestSet = sum(abs((probs_of_positive>threshold)-y_test)) / size(y_test,1);
end








