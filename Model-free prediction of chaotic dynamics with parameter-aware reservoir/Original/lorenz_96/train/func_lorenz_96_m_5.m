function [X,ft] = func_lorenz_96_m_5(A)
m = 5;                  % 节点数目
%A = 2;                  % 驱动幅度
omega = 2;              % 驱动频率
F = 2;                  % 驱动偏移量
tspan = [0 :0.01: 1500];        % 时间范围
initial_conditions = rand(1, m);  % 初始条件，可以根据需要调整
% 使用ode45求解ODE
[t, X] = ode45(@(t, x) lorenz96(t, x, A, omega, F, m), tspan, initial_conditions);

ft = A * sin(omega * t) + F;
% 定义右端项（ODEs）
function dx = lorenz96(t, x, A, omega, F, m)
    dx = zeros(m, 1);
    
    % 计算每个节点的动态
    for i = 1:m
        % 周期性边界条件：获取前后节点
        x_prev = x(mod(i-2, m) + 1);   % x_{i-1}
        x_next = x(mod(i, m) + 1);     % x_{i+1}
        x_before_prev = x(mod(i-3, m) + 1);  % x_{i-2}

        % 驱动项
        f_t = A * sin(omega * t) + F;
        
        % 节点的动力学方程
        dx(i) = x_prev * (x_next - x_before_prev) - x(i) + f_t;
    end
end
end
