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

# 4. 시각화 (CairoMakie)
fig = Figure(size=(1200, 900))

ax1 = Axis(fig[1, 1], title="1. Stable Stripes", xlabel="Time", ylabel="Space")
heatmap!(ax1, t1, 1:N, data1', colormap=:viridis)

ax2 = Axis(fig[1, 2], title="2. SH Turbulence (KS-like)", xlabel="Time", ylabel="Space")
heatmap!(ax2, t2, 1:N, data2', colormap=:magma)

ax3 = Axis(fig[2, 1], title="3. Intermittent State", xlabel="Time", ylabel="Space")
heatmap!(ax3, t3, 1:N, data3', colormap=:thermal)

ax4 = Axis(fig[2, 2], title="4. High-freq Chaos", xlabel="Time", ylabel="Space")
heatmap!(ax4, t4, 1:N, data4', colormap=:inferno)

save("swift_hohenberg_dynamics.png", fig)