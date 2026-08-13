
% Generate traindata (data corresponding to four bifurcation parameters, 2*8000)
% The initial value is randomly assigned to the system, and the logistic data 
% of the last 2000 steps is obtained for each bifurcation parameter a
% The variable time series of each bifurcation parameter the next action is
% the value of the bifurcation parameter (as the channel input data guided 
% by the parameter)
clc;
clear;
totle_time=3000;  % Total iteration time
de_time=1000;     % Transient time
time=totle_time-de_time; % The final retained time step for each bifurcation
                         % parameter used to train RC
a1=3.3;   % Period 2
a2=3.5;   % Period 4
a3=3.6;   % chaos
a4=3.8;   % chaos
D_in=2;   % The number of data variables, the variable x of the logistic map
          % itself and the corresponding bifurcation parameter a
u=zeros(D_in,4*time);    
% u1,u2,u3,u4 are the last 2000 step time series corresponding to the above
% bifurcation parameters, and each matrix is 1*2000
%% Reserve storage space for the time series of system state variables
u1=zeros(1,time); % The time series storage space corresponding to the first parameter a1
u2=zeros(1,time);
u3=zeros(1,time);
u4=zeros(1,time);

% Calculate the time series of the Logistic map corresponding to the different
% bifurcation parameters
data_logistic1=fun_logistic(totle_time,de_time,a1);
data_logistic2=fun_logistic(totle_time,de_time,a2);
data_logistic3=fun_logistic(totle_time,de_time,a3);
data_logistic4=fun_logistic(totle_time,de_time,a4);

u1=data_logistic1;  % u1 holds the last 2000 steps corresponding to the 
                    % first bifurcation parameter a1
u2=data_logistic2;  
u3=data_logistic3; 
u4=data_logistic4; 
% The data corresponding to different bifurcation parameters are strung together
% to form the training data u (the first behavior is x under different parameters,
% second behavior corresponding to the bifurcation parameter), used to train RC
u(1,1:time)=u1;         
u(1,time+1:2*time)=u2; 
u(1,2*time+1:3*time)=u3;
u(1,3*time+1:end)=u4; 
% The second behavior of the training data matrix is the bifurcation parameter
% corresponding to the state vector of the first row
u(2,1:time)=a1;             
u(2,time+1:2*time)=a2; 
u(2,2*time+1:3*time)=a3;
u(2,3*time+1:end)=a4; 
traindata=u;
save('traindata.mat','traindata'); 

%% Logistic map system time series program (function program)
% Input parameters: totle_time (total duration),de_time (removed transient duration),
% a (bifurcation parameter)
function data_log=fun_logistic(totle_time,de_time,a)
    x=zeros(1,totle_time); 
    rng(1);            % random seed
    x(1)=rand(1,1);    % The initial value of x is random between (0,1)
     for t=1:totle_time-1
        x(t+1)=a*x(t)*(1-x(t)); 
     end
    data_log=x(de_time+1:end);  % Retain the data after removing the former
                                % de_time transient time
end
