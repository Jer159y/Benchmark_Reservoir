function stuart_landau!(du, u, p, t)
    N, λ, ω, K = p
    # u is ComplexF64 vector of length N 
    for i in 1:N
        # Individual node dynamics: dz/dt = (λ + iω - |z|²)z
        du[i] = (λ[i] + im*ω[i] - abs2(u[i])) * u[i]
        
        # Linear coupling: Σ Kij * (zj - zi)
        # Matrix operation optimization could replace loops with slicing or matrix multiplication, but original structure is maintained
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
# 3. 시각화 (CairoMakie Subplots)
# ---------------------------------------------------------
fig = Figure(size=(1200, 900))

ax1 = Axis(fig[1, 1], title="1. Amplitude Death", xlabel="Time", ylabel="Oscillator Index")
heatmap!(ax1, t1, 1:N, data1', colormap=:ice)

ax2 = Axis(fig[1, 2], title="2. Phase Sync", xlabel="Time", ylabel="Oscillator Index")
heatmap!(ax2, t2, 1:N, data2', colormap=:viridis)

ax3 = Axis(fig[2, 1], title="3. Chimera Pattern", xlabel="Time", ylabel="Oscillator Index")
heatmap!(ax3, t3, 1:N, data3', colormap=:magma)

ax4 = Axis(fig[2, 2], title="4. Spatiotemporal Chaos", xlabel="Time", ylabel="Oscillator Index")
heatmap!(ax4, t4, 1:N, data4', colormap=:thermal)

save("stuart_landau_dynamics.png", fig)