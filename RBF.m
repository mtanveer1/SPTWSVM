% RBF Kernel

function [k] = RBF(u, v, gamma)

k = exp(-(gamma)*((u-v)*(u-v)'));   
% k = exp(-(0.01)*((u-v)*(u-v)'));   

end

