"""
System: Hindmarsh-Rose Neuron Network. (continuous)
    bursting neuron model with coupling
    node heterogeneity: intrinsic parameters
Equations:
    dx_i/dt = y_i - a_i x_i^3 + b_i x_i^2 - z_i + I_i + Σ_j K_ij (x_j - x_i)
    dy_i/dt = 1 - 5 x_i^2 - y_i
    dz_i/dt = r (s (x_i - x0) - z_i
Arguments:
    du: Derivative vector to be updated.
    u: Current state vector [x_1, y_1, z_1,
                        x_2, y_2, z_2,
                        ...,
                        x_N, y_N, z_N].
    p: Parameters tuple (N, I, K) where:
        N: Number of neurons.
        I: Input current vector.
        K: Coupling matrix.
Returns:
    Updates du in place with computed derivatives.
"""


function hr_net!(du, u, p, t)
    N, input_I, K = p
    # u = [x1, y1, z1, x2, y2, z2, ...]
    for i in 1:N
        idx = 3i - 2
        x, y, z = u[idx], u[idx+1], u[idx+2]
        
        coupling = 0.0
        for j in 1:N
            coupling += K[i,j] * (u[3j-2] - x)
        end
        
        du[idx]   = y - x^3 + 3x^2 - z + input_I[i] + coupling
        du[idx+1] = 1 - 5x^2 - y
        du[idx+2] = 0.006 * (4 * (x + 1.6) - z)
    end
end

function solve_hr(N, p, tspan, u0)
    prob = ODEProblem(hr_net!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-4)
    # 각 시간별로 x_i 값들만 모아서 Matrix 생성
    data = reduce(hcat, [[sol.u[t][3i-2] for i in 1:N] for t in 1:length(sol.t)])'
    return Matrix(data), sol.t
end


N = 3  # number of node
tspan = (0.0, 5000.0)
u0 = randn(3N) * 0.1

# --- Case 1: Spatiotemporal Chaos (불균질한 I + 약한 결합) ---
I_chaos = 2.8 .+ 0.5 * rand(N)
K_chaos = diagm(1 => fill(0.02, N-1), -1 => fill(0.02, N-1))
data1, t1 = solve_hr(N, (N, I_chaos, K_chaos), tspan, u0)

# --- Case 2: Synchronized Bursting (강한 전체 결합) ---
I_sync = fill(3.0, N)
K_sync = fill(0.1 / N, N, N) # 전역 결합
data2, t2 = solve_hr(N, (N, I_sync, K_sync), tspan, u0)

# --- Case 3: Wave Propagation (인접 결합 + 특정 노드 자극) ---
I_wave = fill(2.0, N) # 기본적으로는 조용한 상태
K_wave = zeros(N, N)
for i in 1:N; K_wave[i, mod1(i+1, N)] = 0.2; K_wave[i, mod1(i-1, N)] = 0.2; end
u0_wave = copy(u0)
u0_wave[1] = 2.0 # 첫 번째 뉴런에 강한 자극
data3, t3 = solve_hr(N, (N, I_wave, K_wave), tspan, u0_wave)

# --- Case 4: Chimera-like Intermittency (중간 강도 결합) ---
I_chim = fill(3.2, N)
K_chim = zeros(N, N)
r_range = 10 # 결합 범위
for i in 1:N, j in i-r_range:i+r_range
    if i != j; K_chim[i, mod1(j, N)] = 0.05; end
end
data4, t4 = solve_hr(N, (N, I_chim, K_chim), tspan, u0)


fig = Figure(size=(2500, 900))

ax1 = Axis(fig[1, 1], title="1. Spatiotemporal Chaos", xlabel="Time", ylabel="Neuron Index")
heatmap!(ax1, t1, 1:N, data1', colormap=:magma)

ax2 = Axis(fig[1, 2], title="2. Synchronized Bursting", xlabel="Time", ylabel="Neuron Index")
heatmap!(ax2, t2, 1:N, data2', colormap=:viridis)

ax3 = Axis(fig[2, 1], title="3. Wave Propagation", xlabel="Time", ylabel="Neuron Index")
heatmap!(ax3, t3, 1:N, data3', colormap=:thermal)

ax4 = Axis(fig[2, 2], title="4. Intermittent Pattern", xlabel="Time", ylabel="Neuron Index")
heatmap!(ax4, t4, 1:N, data4', colormap=:plasma)

display(fig)
save("hr_network_dynamics.png", fig)