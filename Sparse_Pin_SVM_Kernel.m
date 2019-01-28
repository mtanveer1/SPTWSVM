%% Authors: Rahul Choudhary & Sanchit Jalan

%--------------Description--------------------
% Function to calculate accuracy, non-zero
% dual variables and training time for
% Sparse Pin SVM for non-linear case.  
%---------------------------------------------


function [accuracy, non_zero_dual_variables, training_time, lambda] = Sparse_Pin_SVM_Kernel(X1_Train, X2_Train, Y_Train, X_Test, Y_Test, c, gamma, epsilon, tau)

%Here c1 = c2 = c
% epsilon1 = epsilon2 = epsilon
% tau1 = tau2 = tau
training_time = 0;
non_zero_dual_variables = 0;

m1 = size(X1_Train, 1);		%Number of samples in 1st class
m2 = size(X2_Train, 1);		%Number of samples in 2nd class
m = m1 + m2;

n = size(X1_Train, 2);

X_Train = [X1_Train; X2_Train];

%-------Sparse Pinball loss SVM classifier--------
K = zeros(size(X_Train, 1), size(X_Train, 1));

for i = 1: size(X_Train, 1)
	for j = 1: size (X_Train, 1)
		K(i, j) = RBF(X_Train(i, :), X_Train(j, :), gamma);
	end
end

H = diag(Y_Train)*K*diag(Y_Train);

Q = [H zeros(m, m); zeros(m, m) zeros(m, m)];
f = [-ones(m, 1); -ones(m, 1)*epsilon];

Aeq = [Y_Train' zeros(1, m)];
Beq = 0;

A = [eye(m) eye(m)];
B = [ones(m, 1)*c];

lb = [ones(m, 1)*(-tau*c); zeros(m, 1)];	%Here gamma belongs to [0, c] and lambda belongs to [-tau*c, c]
ub = [ones(m, 1)*c; ones(m, 1)*c];

tic;
[U, fval] = quadprog(Q, f, A, B, Aeq, Beq, lb, ub);
toc;

training_time = training_time + toc;

lambda = U(1:m, :);
b = 0;
count = 0;

for i = 1: size(lambda, 1)
	if -tau*c < lambda(i) && lambda(i) < c
		K = zeros(size(X_Train, 1), 1);
		for j = 1: size(X_Train, 1) 
			K(j) = RBF(X_Train(j, :), X_Train(i, :), gamma);
		end
		b = b + Y_Train(i) - (lambda.*Y_Train)'*(K);
		count = count + 1;
	end 
end

b = b/count;


%--------Evaluating accuracy of obtained SVM model---------
accuracy = 0;

for i = 1: size(X_Test, 1)
	K = zeros(size(X_Train, 1), 1);
	for j = 1: size(X_Train, 1)
		K(j) = RBF(X_Train(j, :), X_Test(i, :), gamma); 
	end
	if (lambda.*Y_Train)'*(K) + b > 0
		if Y_Test(i) == 1
			accuracy = accuracy + 1;
		end
	else 
		if Y_Test(i) == -1
			accuracy = accuracy + 1;
		end
	end
end

accuracy = accuracy/(size(X_Test, 1));

for i = 1: size(lambda, 1)
	if abs(lambda(i)) > 0.0000001
		non_zero_dual_variables = non_zero_dual_variables + 1;
	end
end

end