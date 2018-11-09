% For adding noise in each feature with zero mean gaussian distribution in the dataset(plrx.mat)

load('plrx.mat');
r=0.5;
lol=data;
[rows,columns]=size(lol);
columns=columns-1;
rows 
columns
for i=1:rows
	for j=1:columns
		mu=0;
		sigma=abs(r*lol(i,j));
		noise = mvnrnd(mu,sigma,1);
		lol(i,j)= lol(i,j) + noise; 
	end
end

lol
data=lol;
save('plrx_0.5.mat', 'data');

% fprintf('\n');
% for i=1:1
% 	for j=1:columns
% 		% fprintf('%u ', lol(i,j));
% 		% lol(i,j)= lol(i,j)*(1 + a*r); 
% 		fprintf('%u ', lol(i,j));
% 	end
% end

