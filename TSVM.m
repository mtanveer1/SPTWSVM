%% Authors: Rahul Choudhary & Sanchit Jalan

%--------------Description--------------------
% Function to calculate accuracy, non-zero
% dual variables and training time for
% TSVM  for linear case.  
%---------------------------------------------


function [accuracy, non_zero_dual_variables, training_time, lambda] = TSVM(X1_Train, X2_Train, X_Test, Y_Test, c)

training_time = 0;
non_zero_dual_variables = zeros(2, 1);

m1 = size(X1_Train, 1);
m2 = size(X2_Train, 1);

n = size(X1_Train, 2);

%---------Original TSVM for 1st class---------
A = X1_Train;
B = X2_Train;
e1 = ones(m1, 1);
e2 = ones(m2, 1);
H = [A e1];
G = [B e2];

delta = 0;

Q = G*inv(H'*H + delta.*eye(n + 1))*G';

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

u = -inv(H'*H + delta.*eye(n + 1))*G'*U;

w1 = u(1:n, :);
b1 = u((n+1):end , :);

for i = 1: size(U, 1)
	if abs(U(i)) > 0.0000001
		non_zero_dual_variables(1) = non_zero_dual_variables(1) + 1;
	end
end


%---------Original TSVM for 2nd class---------
Q = H*inv(G'*G + delta.*eye(n + 1))*H';

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

u = inv(G'*G + delta.*eye(n + 1))*H'*U;

w2 = u(1:n, :);
b2 = u((n+1):end , :);

lambda = U;
for i = 1: size(U, 1)
	if abs(U(i)) > 0.0000001
		non_zero_dual_variables(2) = non_zero_dual_variables(2) + 1;
	end
end


%---------Evaluating accuracy of obtained SVM model---------
id = ones(size(X_Test, 1), 1);

dist1 = abs(X_Test*w1 + b1.*id);
dist2 = abs(X_Test*w2 + b2.*id);

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






























