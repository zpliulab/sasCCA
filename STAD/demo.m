clear
close all
load('MY_STAD_data.mat')
load('rand_idx.mat')
for i = 1:length(X)
    X{i} = (X{i} - min(X{i})) ./ (max(X{i}) - min(X{i}))-0.5;
end
lambda_set = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5];
for la = 1:length(lambda_set)
    lambda = lambda_set(la);
    parfor ci = 1:10
        disp([la, ci])
        [ci_knn_result(la, ci, :), ci_svm_result(la, ci, :)] = opt_pro(X, label, rand_idx(ci, :), lambda);
    end
end
knn_P = ci_knn_result(:,:,1);
knn_R = ci_knn_result(:,:,2);
knn_ACC = ci_knn_result(:,:,3);
knn_F1 = ci_knn_result(:,:,4);
svm_P = ci_svm_result(:,:,1);
svm_R = ci_svm_result(:,:,2);
svm_ACC = ci_svm_result(:,:,3);
svm_F1 = ci_svm_result(:,:,4);

mean_knn_P = mean(knn_P, 2);
mean_knn_R = mean(knn_R, 2);
mean_knn_ACC = mean(knn_ACC, 2);
mean_knn_F1 = mean(knn_F1, 2);
std_knn_P = std(knn_P, 0, 2);
std_knn_R = std(knn_R, 0, 2);
std_knn_ACC = std(knn_ACC, 0, 2);
std_knn_F1 = std(knn_F1, 0, 2);

mean_svm_P = mean(svm_P, 2);
mean_svm_R = mean(svm_R, 2);
mean_svm_ACC = mean(svm_ACC, 2);
mean_svm_F1 = mean(svm_F1, 2);
std_svm_P = std(svm_P, 0, 2);
std_svm_R = std(svm_R, 0, 2);
std_svm_ACC = std(svm_ACC, 0, 2);
std_svm_F1 = std(svm_F1, 0, 2);


%%
function [knn_result, svm_result] = opt_pro(X, label, idx, lambda)
% X: cell; multi-omics dataset
% label: vector; data label.
% idx: matrix; An indicator matrix that divides training and testing data.
% lambda: scalar; regularization parameter.
r = 1/2;
p = 1;
d = 4;
num_view = length(X);
num_tr = round(length(label) * 0.7);
num_te = length(label) - num_tr;
for i = 1:num_view
    X_tr{i} = X{i}(idx(1: num_tr), :);
    X_te{i} = X{i}(idx(num_tr+1: end), :);
end
label_tr = label(idx(1: num_tr));
label_te = label(idx(num_tr+1: end));
% 初始化
G = double(label_tr == label_tr'); sG = diag(sum(G));
delta = ones(1,num_view);
delta = delta / sum(delta);
A = randn(num_tr, d); [U1, ~, V1] = svd(A, 'econ'); A = U1 * V1';
% load initial
dim_tr = zeros(num_view, 1);
D = cell(num_view, 1);
for i = 1:num_view
    dim_tr(i) = size(X_tr{i}, 2);
    D{i} = eye(dim_tr(i));
end

% 求解
for k = 1:500
    for i = 1:num_view
        [W{i}, D{i}] = solveW(A, X_tr{i}, G, sG, lambda, D{i}, p);
    end
    T2 = zeros(num_tr, 1);
    for i = 1:num_view
        T2 = T2 + delta(i) * X_tr{i} * W{i};
    end
    T2 = G * T2;
    [U, ~, V] = svd(T2, 'econ');
    A = U * V';
    for i = 1:num_view
        delta(i) = (trace(A' * sG * A + W{i}' * X_tr{i}' * sG * X_tr{i} * W{i} - 2 * A' * G * X_tr{i} * W{i}) + lambda * norm2p(W{i}, p))^(r-1);
    end
    delta = delta / sum(delta);
    obval(k) = objective(X_tr, W, A, G, sG, delta, lambda);
    if k > 1 && abs(obval(k)-obval(k-1))/obval(k-1) <0.0001
        break
    end
    disp(['Number of iterations: ', num2str(k)])
end
% min(obval),
% figure, plot(obval)
for i = 1:num_view
    W_n{i} = diag(sqrt(W{i} * W{i}'));
    W{i}(W_n{i}<1e-3, :) = 0;
end
%% 测试
% 参数初始化
Te2 = zeros(num_te, 1);
for i = 1:num_view
    Te2 = Te2 - 2 * delta(i) * X_te{i} * W{i};
end
[u, ~, v] = svd(-Te2 / (2 * sum(delta)), 'econ');
A_te = u * v';
Md = fitcknn(A, label_tr, 'NumNeighbors',3);
[predict_label_knn, score, ~] = predict(Md, A_te);

model = svmtrain(label_tr, A, '-t 0 -c 1 -q -b 1');
[predict_label_svm, ~, prob_estimates] = svmpredict(label_te, A_te, model, '-q -b 1');

[knn_P, knn_R, knn_ACC, knn_F1, knn_AUC] = measure(predict_label_knn, label_te, score(:, 2));
[svm_P, svm_R, svm_ACC, svm_F1, svm_AUC] = measure(predict_label_svm, label_te, prob_estimates(:, 2));
knn_result = [knn_P, knn_R, knn_ACC, knn_F1, knn_AUC];
svm_result = [svm_P, svm_R, svm_ACC, svm_F1, svm_AUC];

end

%%
function t = objective(X_tr, W, A, G, sG, delta, lambda)
t= 0;
for i = 1:length(X_tr)
    t =t + delta(i) * (trace(A' * sG * A + W{i}' * X_tr{i}' * sG * X_tr{i} * W{i} - 2 * A' * G * X_tr{i} * W{i}) + lambda * norm2p(W{i}, 1));
end
end
%%
function [W, D] = solveW(A, X, G, sG, lambda, D, p)
for k = 1:1
    W = (2 * X' * sG * X + 2 * lambda * D) \ (2 * X' * G * A);
    D = diag(sqrt(diag(W * W')).^(p - 2));
end
end
function norm2p = norm2p(X, p)
norm2p = sum(sqrt(diag(X * X')).^p);
end
