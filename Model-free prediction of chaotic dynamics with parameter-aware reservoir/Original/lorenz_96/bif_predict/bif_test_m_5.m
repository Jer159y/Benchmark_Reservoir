clc;clear;

load opt_data_A_resize_1000_test_2000_drive_50.mat 
load min_rng_set_A_resize_1000_test_2000_drive_50.mat
load testdata_1.1_0.01_4.mat testdata
%testdata=func_get_testdata();
A_index=1.1:0.01:4;
RC_data = cell(1,length(A_index));

%index = 1:1:50;
% index=[2,36,40,5,10,19,48,20,21,23];
% [testdata]=func_test_data_Kur_para(index);
%%
 result = getfield ( opt_trials,'Fval');
 param= getfield ( opt_trials,'X');
 [sort_result,result_num]=sort(result);
 sort_param=param(result_num,:);
 opt_result=sort_param(1,:);
 
 sort_rng=min_rng_set(result_num);
 opt_rng=sort_rng(1);  
 rng(opt_rng);

drive_num=50;
data_num=train_num;
testnum=train_num;
test_num=train_num;
data_len=ones(1,data_num)*train_length;

%%

resSize =1000; % size of the reservoir nodes;  
initLen = 100;
TrainLen=sum(data_len)-1;
test_Len =20000;
testLen=20000;
inSize = 6; 
outSize = 5;
nonliner_num=2;
X=[];
Yt = train_output_data(1:outSize,2:TrainLen+1);% run the reservoir with the data and collect X
%%

W_in_a = opt_result(2);
Win= (2.0*rand(resSize,inSize)-1.0)*W_in_a;


%%
for i=1:train_num
    indata=train_data{i};
    [X1,W,reg]=func_get_X(inSize,opt_result,opt_rng,indata,Win,train_length);
    X=[X,X1];
end
X=X(:,1:end-1);
%% 
Data_len=[0,data_len];
for i=0:data_num-1
   trainLen=sum(Data_len(1:i+1));
   X(:,trainLen+1-i*initLen:trainLen+initLen-i*initLen)=[];
   Yt(:,trainLen+1-i*initLen:trainLen+initLen-i*initLen)=[];
end
rank=randperm( size(X,2) );  
X=X(:, rank); 
Yt=Yt(:, rank); 
X_T = X';
Wout = Yt*X_T / (X*X_T + reg*eye(nonliner_num*resSize+1));
%%
Y1= zeros(outSize,testLen);
a = opt_result(3);
rmse_dynamic=0;  
rng(opt_rng);

%for i=1:testnum
for i=1:length(A_index)
    Testdata=testdata{i};
    u=Testdata(:,1);
    Y1(:,1)=u(1:5);
    x1=2*rand(resSize,1)-1;
for t = 1:test_Len-1 
    x1 = (1-a)*x1 + a*tanh( Win*u + W*x1 );
    y = Wout*[1;x1;x1.^2;];
    Y1(:,t+1) = y;
     if t<drive_num
     u(1:5)=Testdata(1:5,1+t);
    
    else
    u(1:5) = y;   
     end  
    u(end)=Testdata(end,1);
    %u(end)=1.1;
   
end
RC_data{i}= Y1;
%rmse_dynamic = rmse_dynamic + mean(abs(Y1(:, drive_num+1:testLen) - Testdata(1:5, drive_num+1:testLen)), 'all');
% figure
% % plot(Y1(1,1:testLen),'k.-');
% % hold on;
% plot(Testdata(1,1:testLen),'r.-');
end
rmse_dynamic=rmse_dynamic/testnum;
rmse=rmse_dynamic;
if isnan(rmse) || rmse>10
    rmse=10;
end
save RC_data_A_6_resize_1000_test_2000_drive_50_opt_5_A_0.01.mat RC_data