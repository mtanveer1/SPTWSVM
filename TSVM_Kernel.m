%% Authors: Rahul Choudhary & Sanchit Jalan

%--------------Description--------------------
% Function to calculate accuracy, non-zero
% dual variables and training time for
% TSVM  for non-linear case.  
%---------------------------------------------


function [accuracy, non_zero_dual_variables, training_time, lambda] = TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, c, gamma)

training_time = 0;
non_zero_dual_variables = zeros(2, 1);

m1 = size(X1_Train, 1);
m2 = size(X2_Train, 1);

n = size(X1_Train, 2);

%---------Original TSVM for 1st class---------
A = X1_Train;
B = X2_Train;
C = [A; B];
e1 = ones(m1, 1);
e2 = ones(m2, 1);

S = zeros(m1, m1 + m2);
for i = 1: size(A, 1)
	for j = 1: size (C, 1)
		S(i, j) = RBF(A(i, :), C(j, :), gamma);
	end
end

R = zeros(m2, m1 + m2);
for i = 1: size(B, 1)
	for j = 1: size (C, 1)
		R(i, j) = RBF(B(i, :), C(j, :), gamma);
	end
end

S = [S e1];
R = [R e2];

delta = 0.00000001;

Q = R*inv(S'*S + delta.*eye(m1 + m2 + 1))*R';

f = -e2;
 
Aeq = [];
Beq = [];

Ain = [];
Bin = [];

lb = zeros(m2, 1);
ub = c.*e2;

tic;
[U, fval] = quadprog(Q, f, Ain, Bin, Aeq, Beq, lb, ub);
toc;

training_time = training_time + toc;

z = -inv(S'*S + delta.*eye(m1 + m2 + 1))*R'*U;

u1 = z(1:m1 + m2, :);
b1 = z((m1 + m2 + 1):end , :);

for i = 1: size(U, 1)
	if abs(U(i)) > 0.0000001
		non_zero_dual_variables(1) = non_zero_dual_variables(1) + 1;
	end
end


%---------Original TSVM for 2nd class---------
Q = S*inv(R'*R + delta.*eye(m1 + m2 + 1))*S';

f = -e1;

Aeq = [];
Beq = [];

Ain = [];
Bin = [];

lb = zeros(m1, 1);
ub = c.*e1;

tic;
[U, fval] = quadprog(Q, f, Ain, Bin, Aeq, Beq, lb, ub);
toc;

training_time = training_time + toc;

z = inv(R'*R + delta.*eye(m1 + m2 + 1))*S'*U;

u2 = z(1:m1 + m2, :);
b2 = z((m1 + m2 + 1):end , :);

lambda = U;
for i = 1: size(U, 1)
	if abs(U(i)) > 0.0000001
		non_zero_dual_variables(2) = non_zero_dual_variables(2) + 1;
	end
end


%---------Evaluating accuracy of obtained SVM model---------
id = ones(size(X_Test, 1), 1);

K = zeros(size(X_Test, 1), m1 + m2);

for i = 1: size(X_Test, 1)
	for j = 1: size (C, 1)
		K(i, j) = RBF(X_Test(i, :), C(j, :), gamma);
	end
end

dist1 = abs(K*u1 + b1.*id);
dist2 = abs(K*u2 + b2.*id);

accuracy = 0;

Y_Predicted = zeros(size(X_Test, 1), 1);

for i = 1: size(X_Test, 1)
	if dist1(i) < dist2(i)
		Y_Predicted(i) = 1;				%Class +1 
	else 
		Y_Predicted(i) = -1;			%Class -1
	end
end

accuracy = (sum(Y_Predicted == Y_Test))/(size(Y_Test, 1));

end






























