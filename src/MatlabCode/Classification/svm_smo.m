%% 主程序 - outer loop
function [Apha,b] = svm_smo(x_train,y_train,cost)
% 二分类SVM，注y属于{-1,1}，而不是{0,1}
% 核函数根据自己的需要修改即可
% 需要增加的地方，若要使用时，为保持稀疏性，仅保留非0的alpha，即支持向量
tic
rng(1993)
[m,~] = size(x_train);
global x y c tol apha threshold error_cache non_bound_set % 定义全局变量:x y c tol是常量，apha threshold error_cache non_bound_set是变量
x = x_train; y = y_train; c = cost;    % 输入
apha = zeros(1,m); threshold = 0;      % 参数初始化
error_cache = []; non_bound_set = [];  % 初始化中间变量
tol = 1e-3;                            % 设置KKT条件的tolerance

numChanged = 0; examineAll = 1; epoch = 0; % epoch用于记录迭代步数
while(numChanged > 0 || examineAll)
    numChanged = 0;
    if(examineAll)
        sequence = randperm(m);
        for j = 1:m
            apha_index_j = sequence(j); 
            binary = examinExample(apha_index_j); %%如何检查是否non-bound,如何计算apha_i
            numChanged = numChanged + binary; epoch = epoch + 1;
        end
    else
        num_non_bound = length(non_bound_set); sequence = non_bound_set(randperm(num_non_bound));
        for j = 1:num_non_bound
            apha_index_j = sequence(j);
            binary = examinExample(apha_index_j); 
            numChanged = numChanged + binary; epoch = epoch + 1;
        end
    end
    if(examineAll == 1)
        examineAll = 0;
    elseif(numChanged == 0)
        examineAll =1;
    end            
end
Apha = apha'; b = threshold; 
toc
end

%% eaminExample - inner loop
function binary = examinExample(apha_index_j)
global  x y c apha non_bound_set tol
yj = y(apha_index_j); 
apha_j = apha(apha_index_j); 
Ej = error(apha_index_j); 
rj = Ej * yj; 
KKTviolate = (rj < -tol && apha_j < c) || (rj > tol && apha_j > 0);
if(KKTviolate)  % 检查是否违背KKT条件
    % 首先用choice heuristic找到apha_i
    if(length(non_bound_set) > 1)
        apha_index_i = choice_heuristic(Ej); 
        binary_take_step = take_step(apha_index_i,apha_index_j); 
        if(binary_take_step); binary = 1; return; end
    end
    % 上述操作不成功，则在non-bound_apha中随机找一个，直到试完为止
    sequence = randperm(length(non_bound_set)); 
    for k = 1:length(non_bound_set)
        apha_index_i = sequence(k);
        binary_take_step = take_step(apha_index_i,apha_index_j); 
        if(binary_take_step); binary = 1; return; end
    end
    % 若上述操作仍不能成功，则在所有apha中随机找一个，直到试完为止
    sequence = randperm(size(x,1)); 
    for k = 1:size(x,1)
        apha_index_i = sequence(k);
        binary_take_step = take_step(apha_index_i,apha_index_j); 
        if(binary_take_step); binary = 1; return; end
    end
end
binary = 0; 
end

%% take step
function binary_take_step = take_step(apha_index_i,apha_index_j)
global x y c apha threshold error_cache non_bound_set
if(apha_index_i == apha_index_j); binary_take_step = 0; return; end
apha_i = apha(apha_index_i); apha_j = apha(apha_index_j);
yi = y(apha_index_i); yj = y(apha_index_j); 
Ei = error(apha_index_i); Ej = error(apha_index_j); 
s = yi * yj;
L = max(0,apha_j-apha_i) * (s == -1) + max(0,apha_j+apha_i-c) * (s == 1); 
H = min(c,c+apha_j-apha_i) * (s == -1) + min(c,apha_j+apha_i) * (s == 1); 
if(L == H); binary_take_step = 0; return; end 
xi = x(apha_index_i,:); xj = x(apha_index_j,:); 
kii = kernel(xi,xi); kij = kernel(xi,xj); kjj = kernel(xj,xj); 
eta = kii + kjj - 2*kij;
if(eta > 0)
    aj = apha_j + yj*(Ei-Ej)/eta;
    if(aj < L); aj = L; elseif(aj > H); aj = H; end 
else
    fi = yi*(Ei-threshold) - apha_i*kii - s*apha_j*kij; fj = yj*(Ej-threshold) - s*apha_i*kij - apha_j*kjj;
    Li = apha_i + s*(apha_j-L); Hi = apha_i + s*(apha_j - H);
    Lobj = Li*fi + L*fj + 0.5* Li^2 *kii + 0.5* L^2 *kjj + s*L*Li*kij;
    Hobj = Hi*fi + H*fj + 0.5* Hi^2 *kii + 0.5* H^2 *kjj + s*H*Hi*kij;
    if(Lobj < Hobj - eps)
        aj = L;
    elseif(Lobj > Hobj + eps)
        aj = H;
    else
        aj = apha_j;
    end
end
if(abs(aj-apha_j) < eps*(aj+apha_j+eps))
    binary_take_step = 0; return;
end
ai = apha_i + s*(apha_j - aj); if(ai<0); ai = 0; end
% caculate b1 b2
b1 = threshold - Ei - yi*(ai - apha_i)*kii - yj*(aj - apha_j)*kij;
b2 = threshold - Ej - yi*(ai - apha_i)*kij - yj*(aj - apha_j)*kjj;
b_new = 0.5*(b1+b2);
% 更新 non_bound_set -> threshold
if(ai > 0 && ai < c); non_bound_set = union(non_bound_set,apha_index_i);b_new = b1; else non_bound_set = setdiff(non_bound_set,apha_index_i);end
if(aj > 0 && aj < c); non_bound_set = union(non_bound_set,apha_index_j);b_new = b2; else non_bound_set = setdiff(non_bound_set,apha_index_j);end
if((ai==0 || ai==c) && (aj==0 || aj==c)); b_new = 0.5*(b1+b2); end
% 更新 error_cache
error_cache_new = zeros(length(non_bound_set),2);
for k = 1:length(non_bound_set)   % 注：这里使用length(x)或numel(x)，不要使用size(x,2),因为setdiff等集合操作,会以0×1的size返回空集
    index = non_bound_set(k);
    if(index ~= apha_index_i && index ~= apha_index_j)
        xk = x(index,:); Ek_old = error(index); kik = kernel(xi,xk); kjk = kernel(xj,xk); 
        Ek_new = Ek_old + yi*(ai-apha_i)*kik + yj*(aj - apha_j)*kjk - threshold + b_new;
        error_cache_new(k,:) = [index, Ek_new];
    else
        error_cache_new(k,:) = [index, 0];
    end
end
error_cache = error_cache_new; 
% 更新 参数threshold，apha_i 与 apha_j
threshold = b_new; apha(apha_index_i) = ai; apha(apha_index_j) = aj; 
binary_take_step = 1;
end

%% 计算error: uj - yj
function Error_j = error(apha_index_j) 
global  x y c apha threshold error_cache 
yj = y(apha_index_j); xj = x(apha_index_j,:); 
apha_j = apha(apha_index_j); 
if(apha_j>0 && apha_j<c)
    Error_j = error_cache(error_cache(:,1) == apha_index_j, 2);
else
    uj = 0;
    for k = 1:size(x,1)
        uj = uj + apha(k) * y(k) * kernel(xj,x(k,:));
    end
    uj = uj + threshold;
    Error_j = uj - yj;
end
end


%% 选择第二个apha的heuristic
function apha_index_i = choice_heuristic(Ej)
global error_cache
error = error_cache(:,2); 
[~,I] = max(abs(Ej - error)); 
apha_index_i = error_cache(I,1);
end

%% 核函数
function k_dot = kernel(xi,xj)
k_dot = dot(xi, xj); % 这是线性核
% k_dot = exp((norm(xi-xj))^2 / gamma); % 高斯核
% k_dot = (dot(xi, xj) + a)^d; % 这是多项式核
end






