function adaptive_kuramoto!(du,u,p,t)
"""
System: Adaptive Phase Oscillators. (continuous)
    coupling adaptation, network structure
    node heterogeneity: natural frequencies

Equations:
    dθ_i/dt = ω_i + Σ_j A_ij κ_ij sin(θ_j - θ_i)
    dκ_ij/dt = ε(-κ_ij + cos(θ_i - θ_j))
Arguments:
    du: Derivative vector to be updated.
    u: Current state vector [θ_1, θ_2, ..., θ_N, κ_11, κ_12, ..., κ_NN].
    p: Parameters tuple (N, ω, ε, A) where:
        N: Number of oscillators.
        ω: Natural frequencies vector.
        ε: Adaptation rate.
        A: Adjacency matrix.
Returns:
    Updates du in place with computed derivatives.
"""
    N, ω, ε, A = p
    θ = u[1:N]
    κ = reshape(u[N+1:end], N, N)

    for i in 1:N
        du[i] = ω[i]
        for j in 1:N
            du[i] += A[i,j]*κ[i,j]*sin(θ[j]-θ[i])
        end
    end

    for i in 1:N, j in 1:N
        du[N + (i-1)*N + j] = ε*(-κ[i,j] + cos(θ[i]-θ[j]))
    end
end

function stuart_landau!(du,u,p,t)
"""
Pick 1st!

System: Stuart-Landau Oscillator Network. (continuous)
    complex oscillators with coupling
    node heterogeneity: intrinsic parameters

Equations:
    dz_i/dt = (λ_i + iω_i - |z_i|^2)z_i + Σ_j K_ij (z_j - z_i)
Arguments:
    du: Derivative vector to be updated.
    u: Current state vector [z_1, z_2, ..., z_N] where z_i are complex numbers.
    p: Parameters tuple (N, λ, ω, K) where:
        N: Number of oscillators.
        λ: Growth rates vector.
        ω: Natural frequencies vector.
        K: Coupling matrix.
Returns:
    Updates du in place with computed derivatives.
"""
    N, λ, ω, K = p
    for i in 1:N
        du[i] = (λ[i] + im*ω[i] - abs(u[i])^2)*u[i]
        for j in 1:N
            du[i] += K[i,j]*(u[j]-u[i])
        end
    end
end

function rossler_net!(du,u,p,t)
"""
Pick 4th!

System: Rössler Oscillator Network. (continuous)
    chaotic oscillators with coupling
    node heterogeneity: intrinsic parameters
Equations:
    dx_i/dt = -y_i - z_i + Σ_j K_ij (x_j - x_i)
    dy_i/dt = x_i + a_i y_i
    dz_i/dt = b_i + z_i (x_i - c_i)
Arguments:
    du: Derivative vector to be updated.
    u: Current state vector [x_1, y_1, z_1, x_2, y_2, z_2, ..., x_N, y_N, z_N].
    p: Parameters tuple (N, a, b, c, K) where:
        N: Number of oscillators.
        a: Parameter vector a.
        b: Parameter vector b.
        c: Parameter vector c.
        K: Coupling matrix.
Returns:
    Updates du in place with computed derivatives.
"""
    N, a, b, c, K = p

    for i in 1:N
        xi, yi, zi = u[3i-2], u[3i-1], u[3i]
        du[3i-2] = -yi - zi + sum(K[i,j]*(u[3j-2]-xi) for j in 1:N)
        du[3i-1] = xi + a[i]*yi
        du[3i]   = b[i] + zi*(xi - c[i])
    end
end

function cml_step!(x,r,ε,A)
"""
System: Coupled Map Lattice (CML). (discrete)
    chaotic maps with coupling
    node heterogeneity: map parameters
Arguments:
    x: Current state vector.
    r: Map parameters vector.
    ε: Coupling strength.
    A: Adjacency matrix.
Returns:
    Next state vector after one time step.
"""
    N = length(x)
    fx = r .* x .* (1 .- x)
    return (1-ε).*fx .+ ε .* (A*fx)
end

function HR_net!(du,u,p,t)
"""
Pick 2nd!

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
    N, I, K = p
    for i in 1:N
        x,y,z = u[3i-2], u[3i-1], u[3i]
        du[3i-2] = y - x^3 + 3x^2 - z + I[i] +
                   sum(K[i,j]*(u[3j-2]-x) for j in 1:N)
        du[3i-1] = 1 - 5x^2 - y
        du[3i]   = 0.006*(4*(x+1.6)-z)
    end
end


using DifferentialEquations
using LinearAlgebra
using Plots
using OrdinaryDiffEq

function fhn_net!(du, u, p, t)
    N, a, b, ϵ, I, K = p
    # u_state: 1:N (fast), v_state: N+1:2N (slow)
    U = @view u[1:N]
    V = @view u[N+1:2N]
    dU = @view du[1:N]
    dV = @view du[N+1:2N]

    for i in 1:N
        # Fast variable dynamics + Coupling
        coupling = 0.0
        for j in 1:N
            coupling += K[i,j] * (U[j] - U[i])
        end
        
        dU[i] = U[i] - (U[i]^3)/3 - V[i] + I[i] + coupling
        # Slow variable dynamics
        dV[i] = ϵ * (U[i] + a - b*V[i])
    end
end

# 1. 환경 설정 (2D 격자 구조 네트워크 예시)
M = 20  # 가로 노드 수
N_nodes = M * M
adj = zeros(N_nodes, N_nodes)

# 인접한 노드끼리만 연결 (Nearest Neighbor)
for i in 1:M, j in 1:M
    curr = (i-1)*M + j
    for (di, dj) in [(0,1), (0,-1), (1,0), (-1,0)]
        ni, nj = i+di, j+dj
        if 1 <= ni <= M && 1 <= nj <= M
            adj[curr, (ni-1)*M + nj] = 1.0
        end
    end
end

# 2. 파라미터 설정 (복잡한 패턴 관찰용)
a = 0.7; b = 0.8; ϵ = 0.08
K_strength = 0.1
K = adj * K_strength
input_v = 0.35 .+ 0.1 * rand(N_nodes) # 노드마다 다른 입력 값 (Heterogeneity)

p = (N_nodes, a, b, ϵ, input_v, K)
u0 = randn(2 * N_nodes) * 0.1
tspan = (0.0, 400.0)

# 3. Solver 실행
prob = ODEProblem(fhn_net!, u0, tspan, p)
sol = solve(prob, Tsit5(), reltol=1e-4, saveat=1.0)

# 1. 데이터 추출 및 Flatten
# sol.u는 각 시간대별 상태 벡터의 배열입니다.
# u[1:N_nodes]만 추출하여 시간(rows) x 노드(cols) 행렬로 만듭니다.
time_steps = length(sol.t)
N_nodes = p[1]
data_matrix = zeros(time_steps, N_nodes)

for i in 1:time_steps
    data_matrix[i, :] = sol.u[i][1:N_nodes]
end

# 2. Spatio-temporal Heatmap 그리기
p1 = heatmap(
    1:N_nodes,             # x축: 노드 인덱스
    sol.t,                 # y축: 시간
    data_matrix,           # 값: 상태 u
    xlabel="Node Index",
    ylabel="Time",
    title="Spatio-temporal Dynamics (FHN Network)",
    c=:viridis,            # 컬러맵 (magma, inferno 등 추천)
    clims=(-2.0, 2.0)      # 색상 범위 고정
)

display(p1)
savefig("fhn_spatiotemporal.png")




using DifferentialEquations
using LinearAlgebra
using Plots
using Random

# ---------------------------------------------------------
# 1. 시스템 함수 정의 (ODE용 및 SDE용)
# ---------------------------------------------------------

# 결정론적 역학 (Deterministic part)
function fhn_f!(du, u, p, t)
    N, a, b, ϵ, input_v, K = p
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    # 커플링 연산: K*U - D*U (D는 연결 강도의 합)
    coupling = K * U - (sum(K, dims=2)[:] .* U)
    
    @. du[1:N] = U - (U^3)/3 - V + input_v + coupling
    @. du[N+1:2N] = ϵ * (U + a - b*V)
end

# 확률적 역학 (Stochastic part: Coherence Resonance용 노이즈)
function fhn_g!(du, u, p, t)
    N = p[1]
    σ = 0.12 # 노이즈 강도
    du[1:N] .= σ
    du[N+1:2N] .= 0.0
end

# 데이터 추출 헬퍼 함수
function solve_and_flatten(prob, is_sde=false)
    sol = is_sde ? solve(prob, SOSRI(), saveat=1.0) : solve(prob, Tsit5(), saveat=1.0)
    N = prob.p[1]
    # Time x Node Index 행렬로 변환
    return reduce(hcat, [sol.u[i][1:N] for i in 1:length(sol.t)])', sol.t
end

# ---------------------------------------------------------
# 2. 시뮬레이션 환경 설정
# ---------------------------------------------------------
N = 100
tspan = (0.0, 4000.0)
a, b, ϵ = 0.7, 0.8, 0.08
u0 = randn(2N) * 0.1

# --- Case 1: Spatiotemporal Chaos (불균질한 입력 + 약한 결합) ---
input_chaos = 0.35 .+ 0.15 * rand(N)
K_chaos = diagm(1 => fill(0.05, N-1), -1 => fill(0.05, N-1))
data1, t1 = solve_and_flatten(ODEProblem(fhn_f!, u0, tspan, (N, a, b, ϵ, input_chaos, K_chaos)))

# --- Case 2: Wave Propagation (균일한 입력 + 강한 이웃 결합) ---
input_wave = fill(0.4, N)
K_wave = diagm(1 => fill(0.2, N-1), -1 => fill(0.2, N-1))
data2, t2 = solve_and_flatten(ODEProblem(fhn_f!, u0, tspan, (N, a, b, ϵ, input_wave, K_wave)))

# --- Case 3: Coherence Resonance (임계값 아래 + 노이즈) ---
input_cr = fill(0.25, N) # 발화 임계값 미만
K_cr = zeros(N, N)
data3, t3 = solve_and_flatten(SDEProblem(fhn_f!, fhn_g!, u0, tspan, (N, a, b, ϵ, input_cr, K_cr)), true)

# --- Case 4: Lag Synchronization (단방향 링 결합) ---
input_lag = fill(0.35, N)
K_lag = zeros(N, N); for i in 1:N; K_lag[mod1(i+1, N), i] = 0.15; end
data4, t4 = solve_and_flatten(ODEProblem(fhn_f!, u0, tspan, (N, a, b, ϵ, input_lag, K_lag)))

# --- Case 5: Amplitude Death (극단적 불균질성 + 강한 결합) ---
input_death = range(0.1, 0.6, length=N)
K_death = zeros(N, N); for i in 1:N, j in max(1, i-2):min(N, i+2); K_death[i,j] = 0.6; end
data5, t5 = solve_and_flatten(ODEProblem(fhn_f!, u0, tspan, (N, a, b, ϵ, input_death, K_death)))

# ---------------------------------------------------------
# 3. 통합 시각화 (컬러맵 수정 버전)
# ---------------------------------------------------------
# c=:magma, :viridis, :inferno, :plasma, :thermal 은 거의 모든 환경에서 지원됩니다.

p1 = heatmap(1:N, t1, data1, title="1. Spatiotemporal Chaos", c=:magma)
p2 = heatmap(1:N, t2, data2, title="2. Wave Propagation", c=:inferno)
p3 = heatmap(1:N, t3, data3, title="3. Coherence Resonance", c=:thermal)
p4 = heatmap(1:N, t4, data4, title="4. Lag Synchronization", c=:viridis)
p5 = heatmap(1:N, t5, data5, title="5. Amplitude Death", c=:plasma)

# 빈 공간을 채우기 위한 더미 플롯
p6 = plot(title="FHN Network Benchmark", grid=false, showaxis=false, 
          xticks=false, yticks=false)

layout = @layout [a b c; d e f]
final_plot = plot(p1, p2, p3, p4, p5, p6, layout=layout, size=(1400, 800), 
                  xlabel="Node Index", ylabel="Time")

display(final_plot)
savefig("fhn_all_dynamics_fixed.png")




using DifferentialEquations
using LinearAlgebra
using Plots
using Random

# ---------------------------------------------------------
# 1. HR 시스템 함수 정의 (최적화 버전)
# ---------------------------------------------------------
function hr_net!(du, u, p, t)
    N, input_I, K = p
    # u는 [x1, y1, z1, x2, y2, z2, ...] 형태
    for i in 1:N
        idx = 3i - 2
        x, y, z = u[idx], u[idx+1], u[idx+2]
        
        # 커플링 계산 (x 변수에 대해서만 결합)
        coupling = 0.0
        for j in 1:N
            coupling += K[i,j] * (u[3j-2] - x)
        end
        
        # HR Equations
        du[idx]   = y - x^3 + 3x^2 - z + input_I[i] + coupling
        du[idx+1] = 1 - 5x^2 - y
        du[idx+2] = 0.006 * (4 * (x + 1.6) - z)
    end
end

# 데이터 추출 함수 (x 변수만 추출하여 Flatten)
function solve_hr(N, p, tspan, u0)
    prob = ODEProblem(hr_net!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-4)
    # 각 시간별로 x_i 값들만 모아서 Matrix 생성
    data = reduce(hcat, [[sol.u[t][3i-2] for i in 1:N] for t in 1:length(sol.t)])'
    return data, sol.t
end

# ---------------------------------------------------------
# 2. 파라미터 및 시뮬레이션 설정 (N을 높임)
# ---------------------------------------------------------
N = 200  # 노드 수 상향
tspan = (0.0, 500.0)
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

# ---------------------------------------------------------
# 3. 시각화 (Subplots)
# ---------------------------------------------------------
p1 = heatmap(1:N, t1, data1, title="1. Spatiotemporal Chaos", c=:magma)
p2 = heatmap(1:N, t2, data2, title="2. Synchronized Bursting", c=:viridis)
p3 = heatmap(1:N, t3, data3, title="3. Wave Propagation", c=:thermal)
p4 = heatmap(1:N, t4, data4, title="4. Intermittent Pattern", c=:plasma)

final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900), 
                  xlabel="Neuron Index", ylabel="Time")

display(final_plot)
savefig("hr_network_dynamics.png")


using DifferentialEquations
using LinearAlgebra
using Plots
using Random

# ---------------------------------------------------------
# 1. Stuart-Landau 시스템 함수 (복소수 대응)
# ---------------------------------------------------------
function stuart_landau!(du, u, p, t)
    N, λ, ω, K = p
    # u는 ComplexF64 벡터
    for i in 1:N
        # 개별 노드 역학: dz/dt = (λ + iω - |z|²)z
        du[i] = (λ[i] + im*ω[i] - abs2(u[i])) * u[i]
        
        # 선형 결합: Σ Kij * (zj - zi)
        # 행렬 연산 최적화를 위해 루프 대신 슬라이싱이나 행렬 곱을 쓸 수 있으나 원본 구조 유지
        for j in 1:N
            if K[i,j] != 0
                du[i] += K[i,j] * (u[j] - u[i])
            end
        end
    end
end

# 데이터 추출 함수 (Real part 또는 Phase 추출)
function solve_sl(N, p, tspan, u0)
    prob = ODEProblem(stuart_landau!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)
    # 분석을 위해 복소수의 실수부(Real part)만 추출
    data = reduce(hcat, [real.(sol.u[t]) for t in 1:length(sol.t)])'
    return data, sol.t
end

# ---------------------------------------------------------
# 2. 파라미터 및 현상별 설정 (N=200)
# ---------------------------------------------------------
N = 200
tspan = (0.0, 400.0)
u0 = [exp(im * 2π * rand()) for _ in 1:N] # 초기 위상은 랜덤

# --- Case 1: Amplitude Death (진폭 소멸) ---
# 결합이 매우 강하고 주파수 분산이 클 때 진동이 멈춤
λ1 = fill(0.1, N) 
ω1 = range(-5.0, 5.0, length=N)
K1 = fill(2.0 / N, N, N) 
data1, t1 = solve_sl(N, (N, λ1, ω1, K1), tspan, u0)

# --- Case 2: Phase Synchronization (위상 동기화) ---
# 주파수가 비슷하고 결합이 적당할 때 모든 오실레이터가 정렬
λ2 = fill(1.0, N)
ω2 = 2.0 .+ 0.1 * randn(N)
K2 = fill(0.5 / N, N, N)
data2, t2 = solve_sl(N, (N, λ2, ω2, K2), tspan, u0)

# --- Case 3: Chimera States (비국소적 결합) ---
# 특정 범위의 이웃과만 결합할 때 동기화와 무질서 영역이 공존
λ3 = fill(1.0, N)
ω3 = fill(2.0, N)
K3 = zeros(N, N)
r = 20 # 결합 반경
for i in 1:N, j in i-r:i+r
    if i != j; K3[i, mod1(j, N)] = 0.1; end
end
data3, t3 = solve_sl(N, (N, λ3, ω3, K3), tspan, u0)

# --- Case 4: Spatiotemporal Chaos (위상 난류) ---
# 결합이 약하고 각 노드의 주파수 이질성이 클 때
λ4 = fill(1.0, N)
ω4 = 5.0 * rand(N)
K4 = diagm(1 => fill(0.1, N-1), -1 => fill(0.1, N-1))
data4, t4 = solve_sl(N, (N, λ4, ω4, K4), tspan, u0)

# ---------------------------------------------------------
# 3. 시각화 (Subplots)
# ---------------------------------------------------------
p1 = heatmap(1:N, t1, data1, title="1. Amplitude Death", c=:ice)
p2 = heatmap(1:N, t2, data2, title="2. Phase Sync", c=:viridis)
p3 = heatmap(1:N, t3, data3, title="3. Chimera Pattern", c=:magma)
p4 = heatmap(1:N, t4, data4, title="4. Spatiotemporal Chaos", c=:thermal)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900), xlabel="Oscillator Index", ylabel="Time")
savefig("stuart_landau_dynamics.png")



using DifferentialEquations, LinearAlgebra, Plots, Random

# 1. Swift-Hohenberg 격자 역학 정의
function swift_hohenberg!(du, u, p, t)
    N, r, kc, κ, L = p
    # L은 Laplacian Matrix (2nd derivative)
    # SH의 핵심은 -(L + kc^2*I)^2 u 항임
    
    # (L + kc^2*I) * u 계산
    tmp = L * u + (kc^2) .* u
    # -(L + kc^2*I)^2 * u 계산
    bilaplacian = -(L * tmp + (kc^2) .* tmp)
    
    @. du = r*u - u^3 + κ * bilaplacian
end

# 2. 1D 격자 및 라플라시안 행렬 생성 (Periodic Boundary)
function get_sh_data(N, r, kc, κ, tspan)
    # Laplacian matrix for 1D periodic boundary
    L = Tridiagonal(fill(1.0, N-1), fill(-2.0, N), fill(1.0, N-1))
    L = Array(L)
    L[1, N] = L[N, 1] = 1.0 
    
    u0 = 0.1 * randn(N)
    p = (N, r, kc, κ, L)
    prob = ODEProblem(swift_hohenberg!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)
    
    return reduce(hcat, sol.u)', sol.t
end

# 3. 파라미터 셋업 (다양한 무늬 생성)
N = 400
tspan = (0.0, 1000.0)
kc = 1.0  # 임계 파수 (무늬의 간격을 결정)
κ = 1.0   # 확산 강도

# --- Case 1: Stable Stripes (질서 정연한 무늬) ---
data1, t1 = get_sh_data(N, 0.5, kc, κ, tspan)

# --- Case 2: Spatiotemporal Chaos (KS 시스템과 가장 유사) ---
# r값이 커지면 패턴이 파괴되며 카오스가 발생함
data2, t2 = get_sh_data(N, 2.5, 0.5, 0.05, tspan)

# --- Case 3: Intermittent Turbulence (간헐적 난류) ---
data3, t3 = get_sh_data(N, 0.1, 0.5, 0.5, tspan)

# --- Case 4: High-frequency Chaos (잔물결 카오스) ---
data4, t4 = get_sh_data(N, 2.0, 2.0, 0.2, tspan)

# 4. 시각화
p1 = heatmap(data1, title="1. Stable Stripes", c=:viridis)
p2 = heatmap(data2, title="2. SH Turbulence (KS-like)", c=:magma)
p3 = heatmap(data3, title="3. Intermittent State", c=:thermal)
p4 = heatmap(data4, title="4. High-freq Chaos", c=:inferno)

plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 900), xlabel="Space", ylabel="Time")
savefig("swift_hohenberg_dynamics.png")

using DifferentialEquations, LinearAlgebra, Plots, Random

# ---------------------------------------------------------
# 1. Gray-Scott 1D 시스템 함수 정의
# ---------------------------------------------------------
function gray_scott_1d!(du, u, p, t)
    N, Du, Dv, F, k, L = p
    # L은 1D Laplacian matrix for periodic boundary conditions
    
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    # 확산 항 계산
    diff_U = L * U
    diff_V = L * V
    
    # 반응 항 및 전체 미분 계산
    @. du[1:N]   = Du * diff_U - U * V^2 + F * (1 - U)
    @. du[N+1:2N] = Dv * diff_V + U * V^2 - (F + k) * V
end

# 2. 1D 라플라시안 행렬 생성 (주기적 경계 조건)
function create_1d_laplacian(N)
    L = Tridiagonal(fill(1.0, N-1), fill(-2.0, N), fill(1.0, N-1))
    L = Array(L)
    L[1, N] = 1.0 # Periodic boundary for the first row
    L[N, 1] = 1.0 # Periodic boundary for the last row
    return L
end

# 3. 시뮬레이션 실행 및 데이터 추출 함수
function get_gs_1d_data(N, Du, Dv, F, k, tspan, initial_perturbation=:random)
    L = create_1d_laplacian(N)
    
    u0 = zeros(2N)
    if initial_perturbation == :random
        # 초기 농도를 약간의 노이즈로 설정
        u0[1:N] .= 1.0 .+ 0.01 * randn(N) # U는 1에 가깝게
        u0[N+1:2N] .= 0.01 * randn(N)    # V는 0에 가깝게
    elseif initial_perturbation == :center_V # 중앙에 V를 집중시키는 초기 조건
        u0[1:N] .= 1.0
        u0[N+1:2N] .= 0.0
        center_idx = N ÷ 2
        u0[N+1+max(1, center_idx-5):N+1+min(N, center_idx+5)] .= 0.5
    end

    p = (N, Du, Dv, F, k, L)
    prob = ODEProblem(gray_scott_1d!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-6)
    
    # U 또는 V 중 하나의 농도 변화를 시각화 (여기서는 V)
    return reduce(hcat, [sol.u[t][N+1:2N] for t in 1:length(sol.t)])', sol.t
end

# ---------------------------------------------------------
# 4. 파라미터 셋업 (다양한 1D 패턴)
# ---------------------------------------------------------
N_gs_1d = 300
tspan_gs = (0.0, 3000.0) # 충분히 오래 시뮬레이션해야 패턴이 안정화됩니다.

# --- Case 1: Simple Stripes (반복적인 줄무늬) ---
# F, k 값에 따라 안정적인 줄무늬 패턴
Du1, Dv1, F1, k1 = 0.16, 0.08, 0.035, 0.065
data_gs1, t_gs1 = get_gs_1d_data(N_gs_1d, Du1, Dv1, F1, k1, tspan_gs, :center_V)

# --- Case 2: Soliton-like Patterns (고립파 같은 이동 패턴) ---
# 특정 F, k 값에서 패턴이 생성되고 이동함
Du2, Dv2, F2, k2 = 0.16, 0.08, 0.060, 0.062
data_gs2, t_gs2 = get_gs_1d_data(N_gs_1d, Du2, Dv2, F2, k2, tspan_gs, :center_V)

# --- Case 3: Chaotic Spot Generation (무질서한 점 생성 및 소멸) ---
# 1D에서도 스팟이 생기고 죽는 패턴을 볼 수 있음
Du3, Dv3, F3, k3 = 0.16, 0.08, 0.055, 0.062
data_gs3, t_gs3 = get_gs_1d_data(N_gs_1d, Du3, Dv3, F3, k3, tspan_gs, :center_V)

# ---------------------------------------------------------
# 5. 시각화 (1D Gray-Scott Subplots)
# ---------------------------------------------------------
p1 = heatmap(1:N_gs_1d, t_gs1, data_gs1, title="1. GS 1D Stripes", c=:viridis)
p2 = heatmap(1:N_gs_1d, t_gs2, data_gs2, title="2. GS 1D Solitons", c=:magma)
p3 = heatmap(1:N_gs_1d, t_gs3, data_gs3, title="3. GS 1D Chaotic Spots", c=:thermal)

plot(p1, p2, p3, layout=(1,3), size=(1500, 500), 
     xlabel="Space", ylabel="Time", plot_title="1D Gray-Scott Dynamics")

savefig("gray_scott_1d_dynamics.png")

# ... (using DifferentialEquations, LinearAlgebra, Plots, Random) ...

# 2D Laplacian Matrix (with periodic boundary conditions)
function create_2d_laplacian(M)
    L_1d = Tridiagonal(fill(1.0, M-1), fill(-2.0, M), fill(1.0, M-1))
    L_1d = Array(L_1d)
    L_1d[1, M] = 1.0
    L_1d[M, 1] = 1.0
    
    L_2d = kron(sparse(I, M, M), L_1d) + kron(L_1d, sparse(I, M, M))
    return L_2d # Sparse matrix for efficiency
end

# Gray-Scott 2D Dynamics
function gray_scott_2d!(du, u, p, t)
    M_size, Du, Dv, F, k, L_2d = p
    N = M_size * M_size
    
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    # Diffusion
    diff_U = L_2d * U
    diff_V = L_2d * V
    
    # Reaction
    @. du[1:N]   = Du * diff_U - U * V^2 + F * (1 - U)
    @. du[N+1:2N] = Dv * diff_V + U * V^2 - (F + k) * V
end

# Simulation function for 2D GS
function get_gs_2d_animation(M, Du, Dv, F, k, tspan)
    L_2d = create_2d_laplacian(M)
    N = M*M
    
    u0 = zeros(2N)
    u0[1:N] .= 1.0 # U starts as 1 everywhere
    
    # Perturbation in the center for V
    mid = M ÷ 2
    for i in (mid-5):(mid+5), j in (mid-5):(mid+5)
        idx = (i-1)*M + j
        u0[N+idx] = 0.5 # Small V perturbation in the center
    end
    u0[N+1:2N] .+= 0.01 * randn(N) # Add some random noise to V

    p = (M, Du, Dv, F, k, L_2d)
    prob = ODEProblem(gray_scott_2d!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=10.0, reltol=1e-6) # Slower saveat for animation

    # Animation generation (plotting V concentration)
    anim = @animate for (i, t) in enumerate(sol.t)
        V_state = reshape(sol.u[i][N+1:2N], M, M)
        heatmap(V_state, zlim=(0.0, 1.0), title="GS 2D Pattern (t=$(Int(t)))", 
                c=:grays, aspect_ratio=:equal, axis=false)
    end
    gif(anim, "gs_2d_pattern.gif", fps=10)
end

# ---------------------------------------------------------
# 7. 2D Gray-Scott 시뮬레이션 (애니메이션)
# ---------------------------------------------------------
M_gs_2d = 100 # 100x100 격자
tspan_2d = (0.0, 5000.0)

# 유명한 "Worms" 패턴을 만드는 파라미터
Du_2d, Dv_2d, F_2d, k_2d = 0.16, 0.08, 0.060, 0.062 

# 또는 "Spots" 패턴: Du_2d, Dv_2d, F_2d, k_2d = 0.16, 0.08, 0.035, 0.065
# 또는 "Moving Spots": Du_2d, Dv_2d, F_2d, k_2d = 0.16, 0.08, 0.0545, 0.061

#get_gs_2d_animation(M_gs_2d, Du_2d, Dv_2d, F_2d, k_2d, tspan_2d)
# 이 함수를 주석 해제하고 실행하면 .gif 파일이 생성됩니다.



using DifferentialEquations
using LinearAlgebra
using Plots
using Random

# ---------------------------------------------------------
# 1. HR 시스템 함수 정의 (최적화 버전)
# ---------------------------------------------------------
function hr_net!(du, u, p, t)
    N, input_I, K = p
    # u는 [x1, y1, z1, x2, y2, z2, ...] 형태
    for i in 1:N
        idx = 3i - 2
        x, y, z = u[idx], u[idx+1], u[idx+2]
        
        # 커플링 계산 (x 변수에 대해서만 결합)
        coupling = 0.0
        for j in 1:N
            coupling += K[i,j] * (u[3j-2] - x)
        end
        
        # HR Equations
        du[idx]   = y - x^3 + 3x^2 - z + input_I[i] + coupling
        du[idx+1] = 1 - 5x^2 - y
        du[idx+2] = 0.006 * (4 * (x + 1.6) - z)
    end
end

# 데이터 추출 함수 (x 변수만 추출하여 Flatten)
function solve_hr(N, p, tspan, u0)
    prob = ODEProblem(hr_net!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-4)
    # 각 시간별로 x_i 값들만 모아서 Matrix 생성
    data = reduce(hcat, [[sol.u[t][3i-2] for i in 1:N] for t in 1:length(sol.t)])'
    return data, sol.t
end

# ---------------------------------------------------------
# 2. 파라미터 및 시뮬레이션 설정 (N을 높임)
# ---------------------------------------------------------
N = 200  # 노드 수 상향
tspan = (0.0, 500.0)
u0 = randn(3N) * 0.1

# --- Case 1: Spatiotemporal Chaos (불균질한 I + 약한 결합) ---
I_chaos = 2.8 .+ 0.5 * rand(N)
K_chaos = diagm(1 => fill(0.02, N-1), -1 => fill(0.02, N-1))
data1, t1 = solve_hr(N, (N, I_chaos, K_chaos), tspan, u0)

# --- Case 2: Synchronized Bursting (강한 전체 결합) ---
I_sync = fill(3.0, N)
K_sync = fill(0.1 / N, N, N) # 전역 결합 (평균장 결합)
data2, t2 = solve_hr(N, (N, I_sync, K_sync), tspan, u0)

# --- Case 3: Wave Propagation (인접 결합 + 특정 노드 자극) ---
I_wave = fill(2.0, N) # 기본적으로는 조용한 상태
K_wave = zeros(N, N)
for i in 1:N; K_wave[i, mod1(i+1, N)] = 0.2; K_wave[i, mod1(i-1, N)] = 0.2; end
u0_wave = copy(u0)
u0_wave[1] = 2.0 # 첫 번째 뉴런에 강한 자극
data3, t3 = solve_hr(N, (N, I_wave, K_wave), tspan, u0_wave)

# --- Case 4: Intermittent Pattern (Chimera-like) (중간 강도 결합) ---
I_chim = fill(3.2, N)
K_chim = zeros(N, N)
r_range = 10 # 결합 범위
for i in 1:N, j in i-r_range:i+r_range
    if i != j; K_chim[i, mod1(j, N)] = 0.05; end
end
data4, t4 = solve_hr(N, (N, I_chim, K_chim), tspan, u0)

# --- Case 5: Amplitude Death (매우 강한 결합 + 높은 이질성) ---
# 이미지에는 5번째가 선 그래프로 되어 있으므로, 특정 노드의 시계열 데이터만 추출
I_death = range(2.5, 3.5, length=N) # 노드 간 큰 주파수 편차
K_death = fill(1.0 / N, N, N) # 강한 전역 결합
prob_death = ODEProblem(hr_net!, u0, tspan, (N, I_death, K_death))
sol_death = solve(prob_death, Tsit5(), saveat=1.0, reltol=1e-4)
# 데이터 추출: 여기서는 시계열 그래프를 위해 1번 노드의 x값만 사용
data5_timeseries = [sol_death.u[t][1] for t in 1:length(sol_death.t)]
t5 = sol_death.t

# ---------------------------------------------------------
# 3. 시각화 (Dashboard 스타일)
# ---------------------------------------------------------

# 플롯 테마 설정 (배경을 어둡게, 폰트 색상을 밝게)
default(
    # plot_bgcolor_subplot=:black, # Plots.jl 백엔드에 따라 지원 여부 다름
    # plot_fgcolor_subplot=:white,
    grid=false,
    border=nothing,
    size=(1400, 950), # 전체 이미지 크기
    thickness_scaling=1.2,
    fontfamily="Dejavu Sans Mono", # 터미널 느낌의 폰트
    legend=false,
)

# 1. Spatiotemporal Chaos (Heatmap)
p1 = heatmap(1:N, t1, data1, 
             title="1. Spatiotemporal Chaos", 
             c=:inferno, 
             titlefont=font(14, color=:white), 
             guidefont=font(10, color=:lightgray), 
             tickfont=font(8, color=:lightgray),
             xlabel="Neuron Index", ylabel="Time")

# 2. Synchronized Bursting (Heatmap)
p2 = heatmap(1:N, t2, data2, 
             title="2. Synchronized Bursting", 
             c=:viridis, 
             titlefont=font(14, color=:white), 
             guidefont=font(10, color=:lightgray), 
             tickfont=font(8, color=:lightgray),
             xlabel="Neuron Index", ylabel="Time")

# 3. Wave Propagation (Heatmap)
p3 = heatmap(1:N, t3, data3, 
             title="3. Wave Propagation", 
             c=:grays, # 이미지와 유사한 톤
             titlefont=font(14, color=:white), 
             guidefont=font(10, color=:lightgray), 
             tickfont=font(8, color=:lightgray),
             xlabel="Neuron Index", ylabel="Time")

# 4. Intermittent Pattern (Heatmap)
p4 = heatmap(1:N, t4, data4, 
             title="4. Intermittent Pattern", 
             c=:haline, # 이미지와 유사한 톤
             titlefont=font(14, color=:white), 
             guidefont=font(10, color=:lightgray), 
             tickfont=font(8, color=:lightgray),
             xlabel="Neuron Index", ylabel="Time")

# 5. Amplitude Death (시계열 Plot)
p5 = plot(t5, data5_timeseries, 
          title="5. Amplitude Death", 
          lw=1.5, color=:cyan, 
          titlefont=font(14, color=:white), 
          guidefont=font(10, color=:lightgray), 
          tickfont=font(8, color=:lightgray),
          xlabel="Time", ylabel="x_1") # x1은 첫 번째 뉴런의 x 변수

# 6. 빈 패널 (텍스트 대시보드 타이틀)
p6 = plot(
    title="HR Network Benchmark",
    titlefont=font(20, "Dejavu Sans Mono", :white, halign=:center, valign=:center),
    xlims=(0,1), ylims=(0,1), # 텍스트 위치 고정용
    grid=false, showaxis=false, border=nothing, # 축 및 테두리 제거
    background_color_subplot=:transparent # 투명 배경
)

# 최종 레이아웃 정의 (이미지와 유사하게)
l = @layout [
    a{0.45w} b{0.45w}; 
    c{0.45w} d{0.45w}; 
    e{0.45w} f{0.45w}
]

final_dashboard_plot = plot(p1, p2, p3, p4, p5, p6, layout=l,
                            link=:none, # 각 서브플롯의 축 스케일 독립적으로
                            plot_title="HR Network Benchmark Dashboard", 
                            plot_titlefont=font(24, "Dejavu Sans Mono", :black, halign=:center),
                            background_color_subplot=:white, # 서브플롯 배경을 흰색으로
                            background_color=:white, # 전체 플롯 배경도 흰색으로
                            foreground_color_subplot=:black, # 눈금, 라벨 색상
                            margin=5Plots.mm # 각 플롯 간 여백
)

display(final_dashboard_plot)
savefig("hr_dashboard_plot.png")