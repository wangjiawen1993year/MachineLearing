%% EM�㷨��ʾʵ��������ϸ�˹�ֲ�
% Ů��������̬�ֲ�N(160,2^2)������7000��������������̬�ֲ�N(170,2.5^2)3000��

%% ����������������ֱ��ͼ
% 1/3.5��x�ĸ����ܶȺ���,���ڻ���ֱ��ͼ�İ�����
prob_x = @(x)1/3.5*(0.7*normpdf(x,160,2)+0.3*normpdf(x,170,2.5));
% ������̬�������
rng(1)
x1 = normrnd(160,2,7000,1);
x2 = normrnd(170,2.5,3000,1);
data = [x1; x2];
% ���������Ƶ��ֱ��ͼ
histogram(data,100,'Normalization','probability')
hold on
ezplot(prob_x,[153,178])
title('����x��Ƶ��ֱ��ͼ')
%% �������ϸ���p(x,z)������ͨ����Ҷ˹��ʽ�������ϸ���p(x,z) = p(z)*p(x|z)
prob_z = @(fai,z) fai.^z .* (1-fai).^(1-z);
prob_x_given_z = @(x,z,miu0,sigma0,miu1,sigma1) normpdf(x,miu1,sigma1).^z .* normpdf(x,miu0,sigma0).^(1-z);
JointProb_of_xANDz = @(x,z,fai,miu0,sigma0,miu1,sigma1) prob_z(fai,z) .* prob_x_given_z(x,z,miu0,sigma0,miu1,sigma1);
prob_z_given_x = @(x,z,fai,miu0,sigma0,miu1,sigma1) JointProb_of_xANDz(x,z,fai,miu0,sigma0,miu1,sigma1) ./ ...
    (JointProb_of_xANDz(x,ones(size(x,1),1),fai,miu0,sigma0,miu1,sigma1) + JointProb_of_xANDz(x,zeros(size(x,1),1),fai,miu0,sigma0,miu1,sigma1));

% log-likelihood�����ڼ���loss
loglikelihood = @(x,fai,miu0,sigma0,miu1,sigma1) sum( log(JointProb_of_xANDz(x,ones(size(x,1),1),fai,miu0,sigma0,miu1,sigma1)+ ...
                                                            JointProb_of_xANDz(x,zeros(size(x,1),1),fai,miu0,sigma0,miu1,sigma1)),1);
%% ִ��EM�㷨
x = data;
average_x = mean(x,1); stdev = std(x);
fai = 0.5;
miu0 = average_x; sigma0 = stdev;
miu1 = average_x+1; sigma1 = stdev;
num_of_epoches = 600;
loss_of_EM = zeros(num_of_epoches,1);
for i = 1 : num_of_epoches
    % E����
    Q0 = prob_z_given_x(x,zeros(size(x,1),1),fai,miu0,sigma0,miu1,sigma1);
    Q1 = prob_z_given_x(x,ones(size(x,1),1),fai,miu0,sigma0,miu1,sigma1);
    % M����
    fai = mean(Q1,1);
    miu0 = sum(Q0 .* x,1) / sum(Q0,1);
    miu1 = sum(Q1 .* x,1) / sum(Q1,1);
    sigma0 = sqrt( sum(Q0 .* (x-miu0).*(x-miu0),1) / sum(Q0,1) );
    sigma1 = sqrt( sum(Q1 .* (x-miu1).*(x-miu1),1) / sum(Q1,1) );
    % ��¼ÿ�ֵ�����loss
    loss_of_EM(i) = -loglikelihood(x,fai,miu0,sigma0,miu1,sigma1);
end
figure(2),plot(loss_of_EM(1:30))
title('-log-likelihood')
xlabel('EM����')
display([fai,miu0,sigma0,miu1,sigma1])


