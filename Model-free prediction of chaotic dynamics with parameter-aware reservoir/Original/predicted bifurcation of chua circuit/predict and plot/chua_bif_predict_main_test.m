%  Optimized result 0.0231    0.9720    0.5316    0.0003    0.7477 
% rng=130 , The error is 0.2182 

clc;clear;
load('traindata.mat');
dt=0.001;      % Time step when solving ode equations
mod_num=20;    % Data sampling interval
drive_num=50;  % Drive data time step
load opt_attractor_chua.mat 
load min_rng_set.mat

%%  Bayesian optimization results
result = getfield ( opt_trials,'Fval');
 param= getfield ( opt_trials,'X');
 [sort_result,result_num]=sort(result);
 sort_param=param(result_num,:);
 opt_result=sort_param(1,:);
 sort_rng=min_rng_set(result_num);
 opt_rng=sort_rng(1);  
  rng(opt_rng);


%%  The hyperparameter value is the result of Bayesian optimization

eig_rho =opt_result(1);   % spectral radius of the layer connection matrix
W_in_a = opt_result(2);   % Range of the connection matrix between the input layer 
                          % and the hidden layer
a = opt_result(3);        % Leakage rate
reg = opt_result(4);      % Regularization coefficient
density =opt_result(5);   % The density of the hiddenlayer connection matrix of RC
hyperpara_set=[eig_rho,W_in_a,a,reg,density];   % Hyperparameter set
rng_num=opt_rng;       % random seed
resSize =500;          % size of the reservoir   
initLen = 100;         % Length of data to be removed when training RC
trainLen=length(traindata(1,:))-1;    % The length of input layer data required
                                      % when training RC
inSize = 4;        % Input layer data dimension
outSize = 3;       % Output layer data dimension
nonliner_num=2;    % The RC neuron output function is modeled as a set of 
                    % linear and nonlinear (square terms)
%%
  indata=traindata;  
  X = zeros(nonliner_num*resSize+1,trainLen);  % F(x)=[x; x^2; 1]
  Yt = indata(1:outSize,2:trainLen+1); % Forecast data corresponding to target system data
%%
Win = (2.0*rand(resSize,inSize)-1.0)*W_in_a;
WW = zeros(resSize,resSize);     % Set the storage space for the hidden layer 
                                 % connection matrix before it is processed
for i=1:resSize
    for j=i:resSize
            if (rand()<density)
             WW(i,j)=(2.0*rand()-1.0);
             WW(j,i)=WW(i,j);
            end
    end
end
rhoW = eigs(WW,1);          % Find the maximum eigenvalue
W = WW .* (eig_rho /rhoW);  % Scaling the adjacency matrix
x=2*rand(resSize,1)-1;      % Randomly initializes the state of hidden layer neurons
for t = 1:trainLen
    u = indata(:,t);        % Assign the variable data of training set time step t to u
    x = (1-a)*x + a*tanh( Win*u + W*x );  % Update the status of the RC node
    X(:,t) = [1;x;x.^2;];   % Node state after RC hidden layer neuron output function
end
% Remove the initial step size of the RC prediction data and 
% the actual data of the corresponding target system
    X(:,1:initLen)=[];  % initLen = 100;
   Yt(:,1:initLen)=[];

% Randomizing the sequence of the iterative data of the corresponding variables
% helps to improve the generalization ability of RC prediction and prevent 
% the dependence of RC on time
rank=randperm( size(X,2) );   % Randomizing the sequence of the iterative 
% data of the corresponding variables helps to improve the generalization
% ability of RC prediction and prevent the dependence of RC on time
X=X(:, rank);          % Rearrange the columns of the matrix X using a randomly arranged index
Yt=Yt(:, rank); 
X_T = X';
% Train Wout with randomly scrambled data
Wout = Yt*X_T / (X*X_T + reg*eye(nonliner_num*resSize+1));
%% predict chua circuit bifurcation
bif_chua_pre5=[];  % Reserve predictive bifurcation graph data matrix
% There are two columns in total, the first column is the bifurcation parameter,
% and the second column stores the maximum point corresponding to the bifurcation parameter
r=15.1:0.001:15.7; % Range of bifurcation parameters
n_r=length(r);     % Number of bifurcation parameter
testLen=12000;
y = Wout*[1;x;x.^2;];  % Forecast data of RC
u(1:3,1)=y;            % Returns the output time series of RC to the input 
                       % layer time series channel
for k=1:n_r
    u(4,1)=r(k);       % Add the corresponding bifurcation parameter values
                       % to the control channel
    Y= zeros(outSize,testLen);    
    % Simulation test phase(Warming up phase)
    for t = 1:100      % Run the reservoir for 100 time steps first (transient)
        x= (1-a)*x + a*tanh( Win*u + W*x );
        y = Wout*[1;x;x.^2;];      
        Y(:,t) = y;         % Save the output to the Y matrix
        u(1:3,1) = y;
        u(4,1)=r(k);
    end
    for t = 1:testLen        % Predict the time series of testLen duration
        x= (1-a)*x + a*tanh( Win*u + W*x );
        y = Wout*[1;x;x.^2;];  
        Y(:,t) = y;          
        u(1:3,1) = y;
        u(4,1)=r(k);
    end
    data_chua=Y(1,end-8000+1:end);      
    %% Find the local maximum of the first variable x
    for t=3:length(data_chua(1,:))  % Start at the third point of the first 
                                    % variable x and find the local maximum
        if data_chua(1,t-1)>data_chua(1,t-2)&&data_chua(1,t-1)>data_chua(1,t)
            bif_chua_pre5=[bif_chua_pre5;r(k),data_chua(1,t-1)];
        end
    end
end
save('bif_chua_pre5.mat','bif_chua_pre5');
% load('bif_chua_processed_data.mat');           
% Actual bifurcation diagram data of target system
r_values=[15.15, 15.323, 15.496, 15.67 ];      % Training set bifurcation value sampling point
% figure(1)
% Draw a prediction bifurcation diagram
plot(bif_chua_pre5(:,1),bif_chua_pre5(:,2),'k.','markersize',0.5);
hold on;
% Draw 4 vertical lines and label R-values
for i=1:length(r_values)
    xline(r_values(i), '--b', 'LineWidth', 2); % Use a dashed blue line 
                                               % with a line width of 2
    % Add annotations with x values
    text(r_values(i),0.812, ['r = ' num2str(r_values(i))], 'VerticalAlignment',...
        'bottom', 'HorizontalAlignment', 'right', 'FontSize', 15, 'Color', 'red');
end
% title(' Predicted Chua circuit bifurcation','Fontsize',20,'Color','b');
% Add (a) to the top left corner of the diagram
text(0.01, 0.95, '(a)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');
% axis tight;
xlabel('r','FontName','Times New Roman','FontSize',28);
ylim([0.81 , 1.02]);
ylabel('x','FontName','Times New Roman','FontSize',28);

% figure(2) % Draw the actual bifurcation diagram
% plot(bif_chua_processed_data(:,1),bif_chua_processed_data(:,2),'k.','markersize',0.5);
% hold on;
% for i=1:length
%     xline(r_values(i), '--b', 'LineWidth', 2); 
%     text(r_values(i), 0.812, ['r = ' num2str(r_values(i))], 'VerticalAlignment',...
%         'bottom', 'HorizontalAlignment', 'right', 'FontSize', 12, 'Color', 'red');
% end
% title(' True Chua circuit bifurcation','Fontsize',20,'Color','b');
% text(0.01, 0.95, '(b)', 'Units', 'normalized', 'FontSize', 20, 'FontWeight', 'bold');
% axis tight;
% xlabel('r','FontName','Times New Roman','FontSize',20);
% ylim([0.81 , 1.02]);
% ylabel('x','FontName','Times New Roman','FontSize',20);





