function [k] = RBF(u, v, gamma)

k = exp(-(gamma)*((u-v)*(u-v)'));   

end

