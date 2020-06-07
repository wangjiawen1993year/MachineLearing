% KNN
function [error_rate_TestSet,probs_of_positive] = KNN(k,x_train,y_train,x_test,y_test)
norm_x_train = x_train ./ repmat(sqrt(sum(x_train .* x_train,2)),1,size(x_train,2));
norm_x_test = x_test ./ repmat(sqrt(sum(x_test .* x_test,2)),1,size(x_test,2));
cos_similarity = norm_x_test * norm_x_train';
[sort_distance,I] = sort(cos_similarity,2,'descend');
% 选出k个最近的训练样本
k_neighbor_dist = sort_distance(:,1:k); k_neighbor_index = I(:,1:k); 
probs_of_positive = zeros(size(y_test));
for j = 1 : size(y_test,1)
    probs_of_positive(j) = sum(y_train(k_neighbor_index(j,:)),1) / k;
    % probs_of_positive(j) = sum(y_train(k_neighbor_index(j,:)) .* k_neighbor_dist(j,:)',1) / sum(k_neighbor_dist(j,:));
end
y_pred = probs_of_positive > 0.5;
error_rate_TestSet = sum(xor(y_pred,y_test)) / size(y_test,1);
display(error_rate_TestSet);
end