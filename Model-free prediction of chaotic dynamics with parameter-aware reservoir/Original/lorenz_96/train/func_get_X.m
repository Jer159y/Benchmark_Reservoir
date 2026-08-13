function [X,W,reg]=func_get_X(inSize,hyperpara_set,rng_num,indata,Win,train_length)
rng(rng_num);
eig_rho =hyperpara_set(1);
%W_in_a = hyperpara_set(2);
a = hyperpara_set(3);
reg = hyperpara_set(4);
density =hyperpara_set(5);
resSize =1000; % size of the reservoir nodes;  
%%
%Win = (2.0*rand(resSize,inSize)-1.0)*W_in_a;
WW = zeros(resSize,resSize);
for i=1:resSize
    for j=i:resSize
            if (rand()<density)
             WW(i,j)=(2.0*rand()-1.0);
             WW(j,i)=WW(i,j);
            end
    end
end
rhoW = eigs(WW,1);
W = WW .* (eig_rho /rhoW); 
x = zeros(resSize,1);
for t = 1:train_length
    u = indata(:,t);
    x = (1-a)*x + a*tanh( Win*u + W*x );
    X(:,t) = [1;x;x.^2;];
    %X(:,t) = x;
end