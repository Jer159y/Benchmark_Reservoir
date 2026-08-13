function min_rmse= func_train_repeat_lorenz_96(hyperpara_set,repeat_num,train_data,train_output_data,testdata,train_length,train_num,test_length)
% repeat func_train
%rng(2);
%rmse_set = zeros(repeat_num,1);
%rng_set=zeros(repeat_num,1);
min_rmse=1001;
for repeat_i = 1:repeat_num
    %repeat_i=1;
   % rng_num=repeat_i*20000 + (now*1000-floor(now*1000))*100000;
    rng_num=randi(500);
    rng(rng_num)
    [rmse,rmse_dynamic] = func_train_lorenz_96(hyperpara_set,rng_num,train_data,train_output_data,testdata,train_length,train_num,test_length);
    if rmse<=min_rmse
        min_rmse=rmse;
        min_rng=rng_num;
        min_rmse_dynamic=rmse_dynamic;
    end
end
load min_rng_set.mat min_rng_set
min_rng_set=[min_rng_set,min_rng];
%min_rng_set=zeros(1,iter_max);
%result = getfield ( opt_trials,'Fval');
%save min_rng_set.mat min_rng_set
save min_rng_set.mat min_rng_set
load min_rmse_dynamic_set.mat min_rmse_dynamic_set
min_rmse_dynamic_set=[min_rmse_dynamic_set,min_rmse_dynamic];
save min_rmse_dynamic_set.mat min_rmse_dynamic_set
fprintf('\nrmse_dynamic is %f\n',rmse_dynamic)
fprintf('\nrmse is %f\n',rmse);
