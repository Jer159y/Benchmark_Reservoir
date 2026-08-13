% clc;clear;
load testdata_1.1_0.01_4.mat testdata
load RC_data_1.1_0.01_4_resize_1000_opt_5.mat RC_data
A_values=1.1:0.01:4;

omega =2;
for j = 1:length(A_values)
    A = A_values(j);  % 当前的
    X=RC_data{j};
   % X=testdata{j};
    t=0.05:0.05:0.05*size(X,2);
    % 为了提取周期最大值，我们需要忽略过渡期并确保足够长的时间数据
    t_stable = t(t > 70);  % 只取稳定期之后的数据
   % t_stable=t_stable(:,1:6000);
    X_stable = X(:,t > 70);  % 稳定期之后的状态
    %X_stable=X_stable(:,1:6000);

    % 选择一个时间段来查找每个周期的最大值
    % 这里假设一个周期的时间大约为 2*pi/omega
    period_length = 2*pi/omega;  % 周期长度
    n_periods = floor((t_stable(end) - t_stable(1)) / period_length);  % 周期数

    % 初始化周期最大值存储
    period_max = zeros(n_periods, 1);
%     figure
%     plot(X_stable(1,1:3000),'k.-');
    % 查找每个周期的最大值
    for k = 1:n_periods
        % 计算该周期的时间范围
        t_start = (k-1) * period_length + t_stable(1);
        t_end = k * period_length + t_stable(1);
        
        % 找到该周期内的最大值
        indices = find(t_stable >= t_start & t_stable < t_end);
        period_max(k) = max(X_stable(1,indices));  % 记录x1的最大值
    end
    
    % 存储每个A值下所有周期的最大值
    max_x1{j} = period_max;  % 存储所有周期的最大值
end

% 绘制分岔图
figure;
hold on;
for j = 1:length(A_values)
    plot(repmat(A_values(j), size(max_x1{j})), max_x1{j}, 'k.', 'MarkerSize', 1); % 为每个A值绘制周期最大值
end
ylim([0 1.2]);
hold on;
x = [1.3,2,2.5,3,3.5,3.8];
%x = [1.3,1.5,2,2.5,3,3.3,3.5,3.8];
for i = 1:length(x)
    line([x(i), x(i)], [0, 1.2], 'Color', 'b', 'LineStyle', '--');
end
%title('predict');
xlabel('$\beta$','interpreter','latex');
ylabel('$x_{1,max}$','interpreter','latex');
set(gcf,'unit','centimeters','position',[8 1 8 7]);
 set(gca,'FontName','Times New Roman','FontSize',15);
%grid on;