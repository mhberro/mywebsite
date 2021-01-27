% this script is for 6740 2020summer, HW1 Q3-2

clear; close all;
% rng(3);  % fix the random seed

edges = dlmread('edges.txt');
nodes = readtable('nodes.txt');

label = table2array(nodes(:,3)); % true data labels

%% Adjacency matrix
m = size(label, 1);

A = zeros(m, m); % initializing adjacency matrix
for ii = 1:size(edges, 1)
    A(edges(ii,1), edges(ii,2)) =1;
    A(edges(ii,2), edges(ii,1)) =1;
end

A_copy = A;

%% remove the isolated nodes
% there are three isolated node, hence the updated matrix is 1487x1487

% idx = find(edges(:, 1) == edges(:, 2));
% iso_node = edges(idx,1);    % isolated nodes index

a = sum(A);
[~, iso_node] = find(a == 0);

A(iso_node',:)=[];  % remove rows
A(:,iso_node')=[];  % remove columns

label(iso_node, :)=[]; % remove the corresponding labels.

%% Graph Laplacian
D = diag(sum(A)); % degree matrix
L = D -A;
[U, S] = svd(L);
S = diag(S,0);  % eigenvalues of the Graph-Laplacian

%%
figure;
plot(log10(S),'b-','LineWidth',3);
xlabel('index of $\lambda_i$','Interpreter','Latex','fontsize',14);
ylabel('$\log \lambda_i$','Interpreter','Latex','fontsize',14);
title('log-magnitude of eigenvalue','fontsize',16)
 
     
%% kmeans with eigenvectors of graph Lapacian
%  we consider the eigenvalue 0 if it is less than 1e-6 
T = 1:numel(find(S<=1e-6));  % nullspace size
V = flip(U, 2);
Ut = V(:,T);
label_predict = kmeans(Ut, 2)-1;
acc1 = sum(label_predict == label)/length(label);
acc2 = sum(label_predict == (1-label))/length(label);
accuracy = max(acc1, acc2);

disp(['the classification accuracy: ', num2str(accuracy)]);
