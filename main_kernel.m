%% Authors: Rahul Choudhary  & Sanchit Jalan

%--------------Description--------------------
% This file is used for calculating sparsity 
% (number of non-zero support vectors) in the
% dual problem of SPTWSVM or one of 
% the other two models, i.e., TSVM or Sparse 
% Pin SVM (for the non-linear case). 
%---------------------------------------------


%-------------------------------Loading dataset-------------------------------

%----------The snippet below is used when dataset has a test-train split (uncomment when using)--------------
% load('monks3_train.mat');
% X_Train = data(:, 1: end - 1);
% X1_Train = data(data(:, end) == 1, 1: end - 1);
% X2_Train = data(data(:, end) == -1, 1: end - 1);
% Y_Train = [data(data(:, end) == 1, end); data(data(:, end) == -1, end)];

% load('monks3_test.mat');
% X_Test = data(:, 1: end - 1);
% Y_Test = data(:, end);
%------------------------------------------------------------------------------------------------------------


%-------The snippet below is used when dataset does not have a test-train split(uncomment when using)-------- 
load('sonar.mat');

[M N] = size(data);                       			%Size of original dataset, M are the number of samples and N - 1 are the number of features, last column are the labels

percentage = 50;									%Percentage of samples used for training

m = floor(M*(percentage/100));                      %Total training samples

n = N - 1;                                			%Number of features

x1 = data(data(:, end) == 1, 1: end - 1);			%Samples in data belonging to +1 class
x2 = data(data(:, end) == -1, 1: end - 1);			%Samples in data belonging to -1 class

y1 = data(data(:, end) == 1, end);					%Samples' labels in data belonging to +1 class
y2 = data(data(:, end) == -1, end);					%Samples' labels in data belonging to -1 class

M1 = size(x1, 1);	               					%Total training samples of +1 class
M2 = size(x2, 1);	               					%Total training samples of -1 class

m1 = floor((percentage/100)*M1);                    %No. of Training Samples in +1 class
m2 = floor((percentage/100)*M2);                    %No. of Training Samples of -1 class


X1_Train = x1(1:m1, :);								%Training data for +1 class
X2_Train = x2(1:m2, :);								%Training data for -1 class
Y1_Train = y1(1:m1, :);								%Labels of training data for +1 class 
Y2_Train = y2(1:m2, :);								%Labels of training data for -1 class 

X_Train = [X1_Train; X2_Train];
Y_Train = [Y1_Train; Y2_Train];

X1_Test = x1(m1 + 1: end, :);
X2_Test = x2(m2 + 1: end, :);
Y1_Test = y1(m1 + 1: end, :);
Y2_Test = y2(m2 + 1: end, :);

X_Test = [X1_Test; X2_Test];
Y_Test = [Y1_Test; Y2_Test];
%-----------------------------------------------------------------------------------------------------------





%--------Setting ranges for epsilon, tau, value of c, and gamma---------------------------------------------
epsilon = [0; 0.05; 0.1; 0.2; 0.3; 0.5];			%Here epsilon = epsilon1 (for subproblem 1) = epsilon2 (for subproblem 2)
tau = [0.01; 0.1; 0.2; 0.5; 1.0];					%Here tau = tau1 (for subproblem 1) = tau2 (for subproblem 2)
c = power(10, -6);									%Here c = c1 (for subproblem 1) = c2 (for subproblem 2)
gamma = power(10, -8);								%gamma in RBF kernel, same in both subproblems
%-----------------------------------------------------------------------------------------------------------





%----------------------Model selection(keep only one uncommented when executing)----------------------------

%-------Sparsity for Sparse_Pin_TSVM (uncomment when using)--------
maxx = 0;
for k = 1: 11
	c = c*10;
	gamma = power(10, -8);
	for l = 1: 11
		gamma = gamma*10;
		for i = 1: size(epsilon, 1)
			for j = 1: size(tau, 1)
				[accuracy] = Sparse_Pin_TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, c, gamma, epsilon(i), tau(j));
				if(maxx < accuracy)
					maxx = accuracy;
					finalc = c;
					finalgamma = gamma;
				end
			end
		end
	end
end

ans = [];
sparsity = [];
time = [];
for i = 1: size(epsilon, 1)
	temp = [];
	tempSparsity = [];
	tempTime = 0;
	for j = 1: size(tau, 1)
		[accuracy, non_zero_dual_variables, training_time, lambda] = Sparse_Pin_TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, finalc, finalgamma, epsilon(i), tau(j));
		temp = [temp accuracy];
		if(tau(j) == 0.5)
			tempSparsity = non_zero_dual_variables;
			tempTime = training_time;
		end
	end
	ans = [ans; temp];
	sparsity = [sparsity; tempSparsity'];
	time = [time; tempTime];
end
%------------------------------------------------------------------


%--------Sparsity for Sparse_Pin_SVM (uncomment when using)--------
% maxx = 0;
% for k = 1: 11
% 	c = c*10;
% 	gamma = power(10, -8);
% 	for l = 1: 11
% 		gamma = gamma*10;
% 		for i = 1: size(epsilon, 1)
% 			for j = 1: size(tau, 1)
% 				[accuracy] = Sparse_Pin_SVM_Kernel(X1_Train, X2_Train, Y_Train, X_Test, Y_Test, c, gamma, epsilon(i), tau(j));
% 				if(maxx < accuracy)
% 					maxx = accuracy;
% 					finalc = c;
% 					finalgamma = gamma;
% 				end
% 			end
% 		end
% 	end
% end

% ans = [];
% sparsity = [];
% time = [];
% for i = 1: size(epsilon, 1)
% 	temp = [];
% 	tempSparsity = [];
% 	tempTime = 0;
% 	for j = 1: size(tau, 1)
% 		[accuracy, non_zero_dual_variables, training_time, lambda] = Sparse_Pin_SVM_Kernel(X1_Train, X2_Train, Y_Train, X_Test, Y_Test, finalc, finalgamma, epsilon(i), tau(j));
% 		temp = [temp accuracy];
% 		if(tau(j) == 0.5)
% 			tempSparsity = non_zero_dual_variables;
% 			tempTime = training_time;
% 		end
% 	end
% 	ans = [ans; temp];
% 	sparsity = [sparsity; tempSparsity'];
% 	time = [time; tempTime];
% end
%------------------------------------------------------------------



%-------------Sparsity for TSVM (uncomment when using)-------------
% maxx = 0;
% ans = [];
% sparsity = [];
% time = [];
% c = power(10,-6);
% for k = 1:11
% 	c = c*10;
% 	gamma = power(10, -8);
% 	for l = 1: 11
% 		gamma = gamma*10;
% 		[accuracy, non_zero_dual_variables, training_time] = TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, c, gamma);
% 		if(maxx < accuracy)
% 			maxx = accuracy;
% 			finalc = c;
% 			finalgamma = gamma;
% 		end
% 	end
% end

% [ans, sparsity, time, lambda] = TSVM_Kernel(X1_Train, X2_Train, X_Test, Y_Test, finalc, finalgamma);
% sparsity = sparsity'; 
%------------------------------------------------------------------

%-----------------------------------------------------------------------------------------------------------



ans = 100.*ans;
ans = round(ans, 3);
ans = single(ans);
% disp(ans);

disp(sparsity);

% disp(time);
%-----------------------------------------------------------------------------------------------------------------------------
