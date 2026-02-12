"""
System: FitzHugh-Nagumo Neuron Network. (continuous + stochastic)
    excitable neuron model with coupling
    node heterogeneity: intrinsic parameters
Equations:
    dU_i/dt = U_i - (U_i^3)/3 - V_i + I_i + Σ_j K_ij (U_j - U_i)
    dV_i/dt = ϵ (U_i + a - b V_i)
Arguments:
    du: Derivative vector to be updated.
    u: Current state vector [U_1, U_2, ..., U_N, V_1, V_2, ..., V_N].
    p: Parameters tuple (N, a, b, ϵ, I, K) where:
        N: Number of neurons.
        a, b, ϵ: Intrinsic parameters.
        I: Input current vector.
        K: Coupling matrix.
Returns:
    Updates du in place with computed derivatives.
"""


# Deterministic part of FitzHugh-Nagumo Network
function fhn_f!(du, u, p, t)
    N, a, b, ϵ, input_v, K = p
    U = @view u[1:N]
    V = @view u[N+1:2N]
    
    # Coupling operation: K*U - D*U (D is the sum of connection strengths)
    coupling = K * U - (sum(K, dims=2)[:] .* U)
    
    @. du[1:N] = U - (U^3)/3 - V + input_v + coupling
    @. du[N+1:2N] = ϵ * (U + a - b*V)
end

# Stochastic part: Noise for Coherence Resonance
function fhn_g!(du, u, p, t)
    N = p[1]
    σ = 0.12 # Noise intensity
    du[1:N] .= σ
    du[N+1:2N] .= 0.0
end

# Data extraction helper function
function solve_and_flatten(prob, is_sde=false)
    sol = is_sde ? solve(prob, SOSRI(), saveat=1.0) : solve(prob, Tsit5(), saveat=1.0)
    N = prob.p[1]
    # Convert to Time x Node Index matrix
    return reduce(hcat, [sol.u[i][1:N] for i in 1:length(sol.t)])', sol.t
end

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


fig = Figure(size=(1400, 800))

ax1 = Axis(fig[1, 1], title="1. Spatiotemporal Chaos", xlabel="Time", ylabel="Node Index")
heatmap!(ax1, t1, 1:N, data1', colormap=:magma)

ax2 = Axis(fig[1, 2], title="2. Wave Propagation", xlabel="Time", ylabel="Node Index")
heatmap!(ax2, t2, 1:N, data2', colormap=:inferno)

ax3 = Axis(fig[1, 3], title="3. Coherence Resonance", xlabel="Time", ylabel="Node Index")
heatmap!(ax3, t3, 1:N, data3', colormap=:thermal)

ax4 = Axis(fig[2, 1], title="4. Lag Synchronization", xlabel="Time", ylabel="Node Index")
heatmap!(ax4, t4, 1:N, data4', colormap=:viridis)

ax5 = Axis(fig[2, 2], title="5. Amplitude Death", xlabel="Time", ylabel="Node Index")
heatmap!(ax5, t5, 1:N, data5', colormap=:plasma)

ax6 = Axis(fig[2, 3], title="FHN Network Benchmark")
hidedecorations!(ax6)
hidespines!(ax6)

display(fig)
save("fhn_all_dynamics_fixed.png", fig)
