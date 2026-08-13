
% The hyperparameter set corresponding to the optimal solution is obtained
% by Bayesian optimization
function [min_rmse,Y]= func_train_repeat_attractor_with_lable(hyperpara_set,repeat_num,traindata,testdata)

min_rmse=1001;
for repeat_i = 1:repeat_num
    rng_num=randi(500);
    rng(rng_num)
    [rmse,rmse_dynamic,Y] = func_train_attractor_with_lable(hyperpara_set,traindata,rng_num,testdata);
    if rmse<=min_rmse
        min_rmse=rmse                                                                                                                         ;
        min_rng=rng_num;
        min_rmse_dynamic=rmse_dynamic;
    end
end
load min_rng_set.mat min_rng_set
min_rng_set=[min_rng_set,min_rng];

save min_rng_set.mat min_rng_set
load min_rmse_dynamic_set.mat min_rmse_dynamic_set
min_rmse_dynamic_set=[min_rmse_dynamic_set,min_rmse_dynamic];
save min_rmse_dynamic_set.mat min_rmse_dynamic_set
fprintf('\nrmse_dynamic is %f\n',rmse_dynamic)
fprintf('\nrmse is %f\n',rmse);
