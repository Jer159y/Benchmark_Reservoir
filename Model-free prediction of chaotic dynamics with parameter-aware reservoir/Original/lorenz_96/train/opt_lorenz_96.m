%% config
clc
clear
iter_max = 100;
repeat_num =4; % ensemble average size
% 1~5: eig_rho, W_in_a, a, reg, d.
lb = [0 0 0 10^-10 0];
ub = [3 3 1 10^-2  1];
[train_data,train_output_data,testdata,train_length,train_num,test_length]=func_dataset_lorenz_96();
%testdata=func_gen_dynamic_test_data(A_set1,Y0_set);
options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'PlotFcn','surrogateoptplot');
%options = optimoptions('surrogateopt','MaxFunctionEvaluations',iter_max,'Display','off','PlotFcn',[]);
filename = ['opt_Logi_1_' datestr(now,30) '_' num2str(randi(999)) '.mat'];
min_rmse = @(x) (func_train_repeat_lorenz_96(x,repeat_num,train_data,train_output_data,testdata,train_length,train_num,test_length));
%% main (don't need to change this part)
%rng((now*1000-floor(now*1000))*100000)
tic
[opt_result,opt_fval,opt_exitflag,opt_output,opt_trials] = surrogateopt(min_rmse,lb,ub,options);
toc
save(filename)
if ~ispc
    exit;
end