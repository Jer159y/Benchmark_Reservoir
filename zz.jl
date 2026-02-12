using Plots

function simulate_cml(N, steps, r, ϵ)
    # x: 상태 벡터, r: 맵 파라미터(3.57 이상에서 카오스), ϵ: 결합 강도
    x = rand(N)
    data = zeros(steps, N)
    
    for t in 1:steps
        # 1. Local Dynamics (Logistic Map)
        f_x = r .* x .* (1 .- x)
        
        # 2. Coupling (Nearest Neighbor with Periodic Boundary)
        next_x = copy(f_x)
        for i in 1:N
            im1 = mod1(i-1, N)
            ip1 = mod1(i+1, N)
            next_x[i] = (1-ϵ)*f_x[i] + (ϵ/2)*(f_x[im1] + f_x[ip1])
        end
        x = next_x
        data[t, :] = x
    end
    return data
end

# 실행: r=4.0(완전 카오스), ϵ=0.1
data_cml = simulate_cml(200, 500, 4.0, 0.1)
heatmap(data_cml, c=:viridis, title="Coupled Map Lattice", xlabel="Space", ylabel="Time")


using DifferentialEquations, LinearAlgebra

function fhn_1d!(du, u, p, t)
    N, a, b, ϵ, I, D, L = p
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    # U 변수에만 확산(Laplacian) 적용
    diff_U = D * (L * U)
    
    @. du[1:N] = U - (U^3)/3 - V + I + diff_U
    @. du[N+1:2N] = ϵ * (U + a - b*V)
end

N_fhn = 200
L = Array(Tridiagonal(fill(1.0, N_fhn-1), fill(-2.0, N_fhn), fill(1.0, N_fhn-1)))
L[1, N_fhn] = L[N_fhn, 1] = 1.0 # Periodic Boundary

p = (N_fhn, 0.7, 0.8, 0.08, 0.35, 0.1, L) # D=0.1 (확산 계수)
u0 = randn(2*N_fhn) * 0.1
prob = ODEProblem(fhn_1d!, u0, (0.0, 400.0), p)
sol = solve(prob, Tsit5(), saveat=1.0)

data_fhn = reduce(hcat, [sol.u[i][1:N_fhn] for i in 1:length(sol.t)])'
heatmap(data_fhn, c=:magma, title="FHN 1D Chain")

function fisher_kpp!(du, u, p, t)
    N, r, D, L = p
    # du/dt = r*u*(1-u) + D*∇²u
    diff_u = D * (L * u)
    @. du = r * u * (1 - u) + diff_u
end

N_kpp = 200
L_kpp = Array(Tridiagonal(fill(1.0, N_kpp-1), fill(-2.0, N_kpp), fill(1.0, N_kpp-1)))
L_kpp[1, N_kpp] = L_kpp[N_kpp, 1] = 1.0

# 국소적인 자극에서 시작 (중앙에만 물질 배치)
u0_kpp = zeros(N_kpp)
u0_kpp[95:105] .= 0.8

p_kpp = (N_kpp, 1.0, 0.1, L_kpp)
prob_kpp = ODEProblem(fisher_kpp!, u0_kpp, (0.0, 200.0), p_kpp)
sol_kpp = solve(prob_kpp, Tsit5(), saveat=1.0)

data_kpp = reduce(hcat, sol_kpp.u)'
heatmap(data_kpp, c=:thermal, title="Fisher-KPP Equation")


using DifferentialEquations, LinearAlgebra, Plots, Random

# ---------------------------------------------------------
# SH 시스템 함수 정의
# ---------------------------------------------------------
function swift_hohenberg!(du, u, p, t)
    N, r, kc, κ, L = p
    
    # (L + kc^2*I) * u 계산
    tmp = L * u + (kc^2) .* u
    # -(L + kc^2*I)^2 * u 계산
    bilaplacian = -(L * tmp + (kc^2) .* tmp)
    
    @. du = r*u - u^3 + κ * bilaplacian
end

# 1D 격자 및 라플라시안 행렬 생성 (Periodic Boundary)
function create_1d_laplacian(N)
    L = Tridiagonal(fill(1.0, N-1), fill(-2.0, N), fill(1.0, N-1))
    L = Array(L)
    L[1, N] = L[N, 1] = 1.0 
    return L
end

function get_sh_data(N, r, kc, κ, tspan, u0=nothing)
    L = create_1d_laplacian(N)
    if u0 === nothing; u0 = 0.1 * randn(N); end
    p = (N, r, kc, κ, L)
    prob = ODEProblem(swift_hohenberg!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)
    return reduce(hcat, sol.u)', sol.t
end

# --- 파라미터 셋업 및 시각화 ---
N_sh = 200
tspan_sh = (0.0, 500.0)

# 1. Stable Stripes
data_sh1, t_sh1 = get_sh_data(N_sh, 0.5, 1.0, 1.0, tspan_sh)

# 2. SH Turbulence (KS-like)
data_sh2, t_sh2 = get_sh_data(N_sh, 1.5, 1.0, 0.5, tspan_sh) # r값 높여 Turbulence 유도

# 3. Intermittent State
data_sh3, t_sh3 = get_sh_data(N_sh, 0.1, 0.5, 0.5, tspan_sh)

# 4. High-frequency Chaos
data_sh4, t_sh4 = get_sh_data(N_sh, 2.0, 2.0, 0.2, tspan_sh)

# Plotting SH
p_sh1 = heatmap(data_sh1, title="1. SH: Stable Stripes", c=:viridis, clims=(-1,1))
p_sh2 = heatmap(data_sh2, title="2. SH: Turbulence (KS-like)", c=:magma, clims=(-2,2))
p_sh3 = heatmap(data_sh3, title="3. SH: Intermittent State", c=:thermal, clims=(-0.5,0.5))
p_sh4 = heatmap(data_sh4, title="4. SH: High-freq Chaos", c=:inferno, clims=(-2,2))

plot(p_sh1, p_sh2, p_sh3, p_sh4, layout=(2,2), size=(1200, 900), 
     xlabel="Space", ylabel="Time", plot_title="Swift-Hohenberg Dynamics")

using DifferentialEquations, LinearAlgebra, Plots, Random, ComplexMixtures # ComplexMixtures는 여기서는 필요없지만 복소수 다룰때 유용

# ---------------------------------------------------------
# CGL 시스템 함수 정의 (복소수 u)
# ---------------------------------------------------------
function complex_ginzburg_landau!(du, u, p, t)
    N, μ, β, ν, δ, L = p
    # L은 1D Laplacian matrix
    
    # 확산 항 (Laplacian for complex field)
    diff_u = L * u
    
    @. du = μ * u + β * abs2(u) * u + ν * diff_u + δ * abs2(u) * diff_u
end

function get_cgl_data(N, μ, β, ν, δ, tspan, u0=nothing)
    L = create_1d_laplacian(N) # SH에서 정의한 라플라시안 재활용
    if u0 === nothing; u0 = randn(N) + im*randn(N); end # 초기값을 복소수로
    p = (N, μ, β, ν, δ, L)
    prob = ODEProblem(complex_ginzburg_landau!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)
    # 복소수의 실수부(real part) 또는 진폭(abs)을 시각화
    return reduce(hcat, real.(sol.u))', sol.t 
end

# --- 파라미터 셋업 및 시각화 ---
N_cgl = 200
tspan_cgl = (0.0, 500.0)

# 1. Plane Wave (안정적인 진행파)
data_cgl1, t_cgl1 = get_cgl_data(N_cgl, 0.1, -0.5, 0.1, 0.0, tspan_cgl)

# 2. Phase Turbulence (KS-like 난류)
# 무질서한 위상과 진폭의 요동
data_cgl2, t_cgl2 = get_cgl_data(N_cgl, 0.5, -0.5, 0.1, -0.2, tspan_cgl) 

# 3. Defect Chaos (결함 카오스)
# 위상의 특이점(Defect)이 생기고 사라지는 현상
data_cgl3, t_cgl3 = get_cgl_data(N_cgl, 0.5, -0.5, 0.1, 0.2, tspan_cgl) 

# 4. Amplitude Chaos (진폭 카오스)
data_cgl4, t_cgl4 = get_cgl_data(N_cgl, 1.0, -1.0, 0.1, -0.5, tspan_cgl)

# Plotting CGL
p_cgl1 = heatmap(data_cgl1, title="1. CGL: Plane Wave", c=:viridis, clims=(-1,1))
p_cgl2 = heatmap(data_cgl2, title="2. CGL: Phase Turbulence", c=:magma, clims=(-2,2))
p_cgl3 = heatmap(data_cgl3, title="3. CGL: Defect Chaos", c=:thermal, clims=(-2,2))
p_cgl4 = heatmap(data_cgl4, title="4. CGL: Amplitude Chaos", c=:inferno, clims=(-3,3))

plot(p_cgl1, p_cgl2, p_cgl3, p_cgl4, layout=(2,2), size=(1200, 900), 
     xlabel="Space", ylabel="Time", plot_title="Complex Ginzburg-Landau Dynamics")

# ... (using DifferentialEquations, LinearAlgebra, Plots, Random) ...
using LinearAlgebra
# ---------------------------------------------------------
# LL 시스템 함수 정의 (복소수 u)
# ---------------------------------------------------------
function lugiato_lefever!(du, u, p, t)
    N, θ, α, Δ, F, L = p
    # θ: 감쇠율, α: 비선형 계수, Δ: 디튜닝, F: 외부 구동, L: Laplacian
    
    diff_u = L * u # 확산 항
    
    @. du = -(1 + im*Δ) * u + F + α * abs2(u) * u + im * θ * diff_u
end

function get_ll_data(N, θ, α, Δ, F, tspan, u0=nothing)
    L = create_1d_laplacian(N)
    if u0 === nothing; u0 = rand(N) + im*rand(N); end
    p = (N, θ, α, Δ, F, L)
    prob = ODEProblem(lugiato_lefever!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0)
    return reduce(hcat, abs.(hcat(sol.u...)))', sol.t # 진폭(Amplitude) 시각화
end

# --- 파라미터 셋업 및 시각화 ---
N_ll = 200
tspan_ll = (0.0, 1000.0)

# 1. Stable Homogeneous State (균일한 안정 상태)
data_ll1, t_ll1 = get_ll_data(N_ll, 0.1, 1.0, 0.0, 1.0, tspan_ll)

# 2. Soliton Formation (솔리톤 형성)
# 특정 파라미터에서 빛의 펄스(솔리톤)가 형성됨
data_ll2, t_ll2 = get_ll_data(N_ll, 0.1, 1.0, 2.0, 1.0, tspan_ll)

# 3. Soliton Gas / Spatiotemporal Chaos (솔리톤 가스 / 시공간 카오스)
# 여러 솔리톤이 상호작용하며 복잡한 패턴 형성
data_ll3, t_ll3 = get_ll_data(N_ll, 0.1, 1.0, 4.0, 1.0, tspan_ll)

# 4. Moving Front (이동하는 전선)
data_ll4, t_ll4 = get_ll_data(N_ll, 0.1, 1.0, -1.0, 1.0, tspan_ll)


# Plotting LL
p_ll1 = heatmap(data_ll1, title="1. LL: Stable State", c=:viridis, clims=(0,2))
p_ll2 = heatmap(data_ll2, title="2. LL: Soliton Formation", c=:magma, clims=(0,5))
p_ll3 = heatmap(data_ll3, title="3. LL: Soliton Gas Chaos", c=:thermal, clims=(0,5))
p_ll4 = heatmap(data_ll4, title="4. LL: Moving Front", c=:inferno, clims=(0,2))

plot(p_ll1, p_ll2, p_ll3, p_ll4, layout=(2,2), size=(1200, 900), 
     xlabel="Space", ylabel="Time", plot_title="Lugiato-Lefever Dynamics")


     # ... (using DifferentialEquations, LinearAlgebra, Plots, Random) ...

# ---------------------------------------------------------
# Gray-Scott 1D 시스템 함수 정의
# ---------------------------------------------------------
function gray_scott_1d!(du, u, p, t)
    N, Du, Dv, F, k, L = p
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    diff_U = L * U
    diff_V = L * V
    
    @. du[1:N]   = Du * diff_U - U * V^2 + F * (1 - U)
    @. du[N+1:2N] = Dv * diff_V + U * V^2 - (F + k) * V
end

# 1D 라플라시안 행렬 생성 (주기적 경계 조건)
# create_1d_laplacian 함수는 SH에서 재활용 가능

function get_gs_1d_data(N, Du, Dv, F, k, tspan, initial_perturbation=:center_V)
    L = create_1d_laplacian(N)
    u0 = zeros(2N)
    
    u0[1:N] .= 1.0 # U는 1로 시작
    if initial_perturbation == :random
        u0[N+1:2N] .= 0.01 * randn(N) # V는 노이즈
    elseif initial_perturbation == :center_V
        center_idx = N ÷ 2
        u0[N+1+max(1, center_idx-5):N+1+min(N, center_idx+5)] .= 0.5 # 중앙에 V 집중
    end

    p = (N, Du, Dv, F, k, L)
    prob = ODEProblem(gray_scott_1d!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=1.0, reltol=1e-6)
    
    # V 농도 변화를 시각화
    return reduce(hcat, [sol.u[t][N+1:2N] for t in 1:length(sol.t)])', sol.t
end

# --- 파라미터 셋업 및 시각화 ---
N_gs = 200
tspan_gs = (0.0, 3000.0) # 충분히 오래 시뮬레이션해야 패턴이 안정화됩니다.

# 1. GS 1D Stripes
Du1, Dv1, F1, k1 = 0.16, 0.08, 0.035, 0.065
data_gs1, t_gs1 = get_gs_1d_data(N_gs, Du1, Dv1, F1, k1, tspan_gs, :center_V)

# 2. GS 1D Solitons (고립파 같은 이동 패턴)
Du2, Dv2, F2, k2 = 0.16, 0.08, 0.060, 0.062
data_gs2, t_gs2 = get_gs_1d_data(N_gs, Du2, Dv2, F2, k2, tspan_gs, :center_V)

# 3. GS 1D Chaotic Spots (무질서한 점 생성 및 소멸)
Du3, Dv3, F3, k3 = 0.16, 0.08, 0.055, 0.062
data_gs3, t_gs3 = get_gs_1d_data(N_gs, Du3, Dv3, F3, k3, tspan_gs, :center_V)

# 4. GS 1D Worms (벌레 모양 패턴)
Du4, Dv4, F4, k4 = 0.16, 0.08, 0.060, 0.058
data_gs4, t_gs4 = get_gs_1d_data(N_gs, Du4, Dv4, F4, k4, tspan_gs, :center_V)

# Plotting GS
p_gs1 = heatmap(data_gs1, title="1. GS 1D: Stripes", c=:viridis, clims=(0,1))
p_gs2 = heatmap(data_gs2, title="2. GS 1D: Solitons", c=:magma, clims=(0,1))
p_gs3 = heatmap(data_gs3, title="3. GS 1D: Chaotic Spots", c=:thermal, clims=(0,1))
p_gs4 = heatmap(data_gs4, title="4. GS 1D: Worms", c=:inferno, clims=(0,1))

plot(p_gs1, p_gs2, p_gs3, p_gs4, layout=(2,2), size=(1200, 900), 
     xlabel="Space", ylabel="Time", plot_title="Gray-Scott 1D Dynamics")