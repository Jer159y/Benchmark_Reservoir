
%clc;clear
function testdata=func_get_testdata()
m=5;
train_length=6000;
test_length=20000;
A_index=[1.3,2,2.5,3,3.5,3.8];
train_num=size(A_index,2);
Train_data=cell(1,train_num);
Testdata=cell(1,train_num);
interval=5;

train_output_data=zeros(2,train_num*train_length);
all_series_data = [];

for j=1:train_num
    A = A_index(j);
    train_data=zeros(m+1,train_length);
    testdata=zeros(m+1,test_length);
    [series_data,ft] = func_lorenz_96_m_5(A);
    series_data=series_data';
    series_data(:,1:10000)=[];  % Trimming the first 10000 data points
    ft(1:10000)=[];

    % Collecting all series data for global normalization
    all_series_data = [all_series_data, series_data];
end

% Calculate global min_x and max_x across all series_data
[min_x, max_x] = bounds(all_series_data(:));
A_index=1.1:0.001:4;
% Now, process each dataset and normalize globally
for j=1:length(A_index)
    A = A_index(j);
    train_data=zeros(m+1,train_length);
    testdata=zeros(m+1,test_length);
    [series_data,ft] = func_lorenz_96_m_5(A);
    series_data=series_data';
    series_data(:,1:10000)=[];  % Trimming the first 10000 data points
    ft(1:10000)=[];

    for i = 1:size(series_data, 1)
        % Downsample series data
        sampled_series(i,:) = downsample(series_data(i,:), interval);

        % Apply global normalization
        normalized_data(i,:) = 2 * (sampled_series(i,:) - min_x) / (max_x - min_x) - 1;

        %train_data(i,:) = normalized_data(i,1:train_length);
        testdata(i,:) = normalized_data(i,train_length+1:train_length+test_length);
       
        %train_output_data(i,(j-1)*train_length+1:j*train_length) = train_data(i,:);
    end 
        sampled_ft = downsample(ft, interval);

        train_data(i+1,:)= A*ones(1,train_length);
        %train_data(i+1,:)= ft(1:train_length);

        testdata(i+1,:) = A*ones(1,test_length);
        %test_data(i+1,:)= ft(1:test_length);

%      figure
%         plot(train_data(1,:),'r.-');
    % Store the data in the corresponding cell
   % Train_data{j} = train_data;
    Testdata{j} = testdata;
end

%train_data = Train_data;
testdata = Testdata;
%save testdata_1.1_0.001_4_length_20000.mat testdata
