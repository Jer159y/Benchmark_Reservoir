using DifferentialEquations
using FFTW
using Plots
using LinearAlgebra
using Random
using Statistics

# 공통 설정: 플롯 테마
default(
    fontfamily="Dejavu Sans Mono",
    titlefont=font(10),
    guidefont=font(8),
    tickfont=font(7),
    margin=3Plots.mm
)

# =========================================================
# 1. Kuramoto-Sivashinsky (KS) Equation
# 문헌: Pathak et al. (PRL 2018)
# 방식: ETDRK4 (Exponential Time Differencing) 대신 
#       DiffEq.jl의 자동 스티프니스 처리를 믿고 Spectral ODE로 구현
# =========================================================
function ks_spectral!(du, u, p, t)
    # u is in Fourier space (coefficient vector)
    k, L = p
    
    # 1. Linear part (hyper-viscosity + negative diffusion): (k^2 - k^4) * u_hat
    # Note: real space eqn is u_t = -u*u_x - u_xx - u_xxxx
    # In Fourier: u_hat_t = -F(u*u_x) + (k^2 - k^4)*u_hat
    
    # Nonlinear part: -u * u_x
    # Transform back to real space to calculate product
    u_real = real(ifft(u))
    ux_real = real(ifft(im .* k .* u))
    nonlinear_term = -u_real .* ux_real
    
    # Transform nonlinear term back to Fourier space
    nonlinear_hat = fft(nonlinear_term)
    
    # Linear part handled implicitly/explicitly or added here
    # Here we put everything in f for simplicity with generic solvers
    @. du = (k^2 - k^4) * u + nonlinear_hat
end

function run_ks()
    N = 64
    L = 22.0 # Chaos domain size
    dx = L/N
    x = range(0, L-dx, length=N)
    k = [0:N/2-1; -N/2:-1] * (2π/L) # Wave numbers
    
    u0_real = cos.(x) .+ 0.1 .* sin.(x./8) .* (1 .+ 0.1.*rand(N))
    u0_hat = fft(u0_real)
    
    tspan = (0.0, 200.0)
    prob = ODEProblem(ks_spectral!, u0_hat, tspan, (k, L))
    sol = solve(prob, Tsit5(), saveat=0.5)
    
    # Convert back to real space for plotting
    data = reduce(hcat, [real(ifft(sol.u[i])) for i in 1:length(sol.t)])'
    heatmap(x, sol.t, data, title="1. KS: Spatiotemporal Chaos", c=:magma, xlabel="x", ylabel="t")
end

# =========================================================
# 2. Multiscale Lorenz-96 (Two-scale)
# 문헌: Chattopadhyay et al. (2020)
# =========================================================
function lorenz96_multiscale!(du, u, p, t)
    # K: Slow vars, J: Fast vars per slow var
    K, J, F, h, c, b = p
    
    # Unpack state
    X = @view u[1:K]
    Y = @view u[K+1:end]
    dX = @view du[1:K]
    dY = @view du[K+1:end]
    
    # Slow variables X dynamics
    for k in 1:K
        kp1 = mod1(k+1, K)
        km1 = mod1(k-1, K)
        km2 = mod1(k-2, K)
        
        # Coupling term from fast variables
        coupling = 0.0
        for j in 1:J
            coupling += Y[(k-1)*J + j]
        end
        
        dX[k] = -X[km1] * (X[km2] - X[kp1]) - X[k] + F - (h*c/b) * coupling
    end
    
    # Fast variables Y dynamics
    for k in 1:K
        for j in 1:J
            idx = (k-1)*J + j
            jp1 = mod1(j+1, J) + (k-1)*J # Simplified periodic bc for Y within block
            jm1 = mod1(j-1, J) + (k-1)*J
            jm2 = mod1(j-2, J) + (k-1)*J
            
            # Proper global indexing for Y needs modulo logic over K*J, 
            # but usually Y is locally coupled or globally ring. 
            # Here we assume global ring for Y.
            idx_p1 = mod1(idx+1, K*J)
            idx_m2 = mod1(idx-2, K*J)
            idx_m1 = mod1(idx-1, K*J)

            dY[idx] = -c * b * Y[idx_p1] * (Y[idx_m2] - Y[idx_m1]) - c * Y[idx] + (h*c/b) * X[k]
        end
    end
end

function run_l96_multi()
    K = 8    # Slow variables
    J = 32   # Fast variables per slow variable
    F = 10.0 # Forcing (적당한 카오스)
    h = 1.0  # Coupling strength
    c = 4.0  # Time scale ratio (낮춤)
    b = 10.0 # Amplitude ratio
    
    # 초기조건 스케일 조정
    u0 = zeros(K + K*J)
    u0[1:K] .= F .+ 0.01 .* randn(K)  # X는 F 근처에서 시작
    u0[K+1:end] .= 0.01 .* randn(K*J)  # Y는 작은 값으로 시작
    
    tspan = (0.0, 10.0)
    
    prob = ODEProblem(lorenz96_multiscale!, u0, tspan, (K, J, F, h, c, b))
    sol = solve(prob, Rodas5(), saveat=0.05, abstol=1e-8, reltol=1e-6)  # Stiff 솔버 사용
    
    # Visualization: Heatmap of X and a sample of Y
    X_data = reduce(hcat, [sol.u[i][1:K] for i in 1:length(sol.t)])'
    Y_data = [sol.u[i][K+1] for i in 1:length(sol.t)]
    
    # NaN 체크 및 스케일 조정
    ylims_val = isempty(filter(!isnan, Y_data)) ? (-1, 1) : extrema(filter(!isnan, Y_data))
    
    p1 = plot(sol.t, X_data[:, 1], label="Slow X1", title="2. L96 Multiscale", lw=2)
    plot!(p1, sol.t, Y_data, label="Fast Y1,1", alpha=0.6)
    return p1
end

# =========================================================
# 3. Complex Ginzburg-Landau (CGL) 1D
# 문헌: Doan et al. (2020)
# =========================================================
function cgl_1d_spectral!(du, u, p, t)
    # u is in Fourier space
    k, μ, β, ν, δ = p
    
    # Linear part: (μ - ν*k^2) * u_hat
    
    # Nonlinear part: (β - i*δ*k^2?? No, δ is usually associated with |A|^2 term spatial derivative or similar)
    # Standard CGL: A_t = A + (1+ib)A_xx - (1+ic)|A|^2 A
    # Mapping to params: μ=1, ν=(1+ib), Nonlin=-(1+ic)
    # Let's use the parameters passed: 
    # du = μ*u + ν*diff_u + β*|u|^2*u + ...
    
    # Calculating nonlinear term in real space
    u_real = ifft(u)
    nonlinear_term = β .* abs2.(u_real) .* u_real
    
    nonlinear_hat = fft(nonlinear_term)
    
    # Combine
    @. du = (μ - ν * k^2) * u + nonlinear_hat
end

function run_cgl()
    N = 128
    L = 50.0
    x = range(0, L, length=N)
    k = [0:N/2-1; -N/2:-1] * (2π/L)
    
    # Defect Turbulence Parameters
    μ = 0.2
    ν = 1.0 + 1.0im # Diffusion coefficient (complex)
    β = -1.0 - 2.0im # Nonlinear coefficient (complex)
    δ = 0.0 # Standard CGL
    
    u0 = 0.1 .* (rand(N) .+ im.*rand(N))
    u0_hat = fft(u0)
    
    tspan = (0.0, 200.0)
    prob = ODEProblem(cgl_1d_spectral!, u0_hat, tspan, (k, μ, β, ν, δ))
    sol = solve(prob, Tsit5(), saveat=1.0)
    
    # Plot Amplitude |A|
    data = reduce(hcat, [abs.(ifft(sol.u[i])) for i in 1:length(sol.t)])'
    heatmap(x, sol.t, data, title="3. CGL: Defect Turbulence", c=:viridis, xlabel="x", ylabel="t")
end

# =========================================================
# 4. Kolmogorov Flow (2D Navier-Stokes)
# 문헌: Canaday et al. (2021)
# Method: 2D Pseudospectral Vorticity-Streamfunction
# =========================================================
function kolmogorov_2d!(du, u, p, t)
    # u is Vorticity (ω) in Fourier space
    Kx, Ky, K2, Re, n = p
    
    # 1. Invert Laplacian to get Streamfunction (ψ)
    # ∇^2 ψ = -ω  =>  ψ_hat = ω_hat / k^2
    psi_hat = -u ./ (K2 .+ 1e-9) # Avoid div/0 at k=0
    psi_hat[1,1] = 0.0
    
    # 2. Compute derivatives for advection term (u⋅∇ω)
    # u_vel = ∂ψ/∂y, v_vel = -∂ψ/∂x
    u_hat = im .* Ky .* psi_hat
    v_hat = -im .* Kx .* psi_hat
    
    omega_x_hat = im .* Kx .* u
    omega_y_hat = im .* Ky .* u
    
    # 3. Compute Nonlinear Term in Real Space (De-aliasing omitted for brevity)
    u_vel = real(ifft(u_hat))
    v_vel = real(ifft(v_hat))
    omega_x = real(ifft(omega_x_hat))
    omega_y = real(ifft(omega_y_hat))
    
    advection = u_vel .* omega_x .+ v_vel .* omega_y
    advection_hat = fft(advection)
    
    # 4. Forcing F = n * cos(n * y)
    # Defined in Fourier space directly for efficiency
    # But here we assume simple forcing in real space for readability
    # F(y) = sin(n*y) is standard
    # F_hat is constant if static, but let's compute explicit linear part
    
    # 5. Combine: ω_t = - (u⋅∇ω) + (1/Re)∇^2ω + F
    # Linear part: -(1/Re)*k^2 * ω_hat
    # Forcing: We add forcing only to specific wavenumber (0, n)
    
    @. du = -(1/Re)*K2 * u - advection_hat
    
    # Add Forcing (Simulated as explicit input to mode (0, n))
    # Grid index for kx=0, ky=n needs care. simplified:
    # We assume F is implicitly handled or added to the specific mode index manually
    # For this demo, let's add a forcing term in real space then FFT
    # (Inefficient but clear)
    # F_real = sin.(n .* grid_y) -> added in setup or loop?
    # Let's add forcing to the specific Fourier mode index directly:
    # Index for (kx=0, ky=n)
    # This is tricky without grid passing. 
    # Let's use real space forcing for simplicity of code.
end

# Wrapper to handle the forcing properly
function kolmogorov_wrapper!(du, u, p, t)
    forcing_hat = p[end] # Precomputed forcing
    params = p[1:end-1]
    kolmogorov_2d!(du, u, params, t)
    @. du += forcing_hat
end

function run_kolmogorov()
    N = 32 # Low res for speed (Use 64+ for real research)
    Re = 40.0 # Turbulent regime
    n = 4     # Forcing wavenumber
    
    L = 2π
    k = [0:N/2-1; -N/2:-1]
    Kx = [i for i in k, j in k]
    Ky = [j for i in k, j in k]
    K2 = Kx.^2 .+ Ky.^2
    
    # Forcing: F = sin(n*y)
    y = range(0, L, length=N)
    Y_grid = [yi for xi in y, yi in y]
    F_real = sin.(n .* Y_grid)
    F_hat = fft(F_real)
    
    # Initial Condition: Random vorticity
    w0 = randn(N, N)
    w0_hat = fft(w0)
    
    tspan = (0.0, 50.0) # Short run
    prob = ODEProblem(kolmogorov_wrapper!, w0_hat, tspan, (Kx, Ky, K2, Re, n, F_hat))
    sol = solve(prob, Tsit5(), saveat=1.0)
    
    # Plot final vorticity field
    w_final = real(ifft(sol.u[end]))
    heatmap(w_final, title="4. Kolmogorov Flow (Vorticity)", c=:balance, aspect_ratio=1)
end

# =========================================================
# 5. Quasi-Geostrophic (QG) Ocean Model (Barotropic)
# 문헌: Srinivasan et al. (2023)
# =========================================================
function qg_barotropic!(du, u, p, t)
    # u is Potential Vorticity (q) in Fourier space
    # q = ∇^2 ψ - F*ψ (if F>0) or just ∇^2 ψ (Barotropic) + βy
    # Here we simulate: q_t + J(ψ, q) = -r*∇^2ψ + ... (Dissipation)
    # q = ∇^2 ψ
    
    Kx, Ky, K2, β, r, ν = p
    
    # 1. Invert q to get ψ
    # ψ_hat = -q_hat / k^2
    psi_hat = -u ./ (K2 .+ 1e-9)
    psi_hat[1,1] = 0.0
    
    # 2. Advection J(ψ, q) = u*q_x + v*q_y
    # u = -ψ_y, v = ψ_x (Geostrophic velocity)
    u_vel_hat = -im .* Ky .* psi_hat
    v_vel_hat = im .* Kx .* psi_hat
    
    q_x_hat = im .* Kx .* u
    q_y_hat = im .* Ky .* u
    
    # Real space multiplication
    u_vel = real(ifft(u_vel_hat))
    v_vel = real(ifft(v_vel_hat))
    q_x = real(ifft(q_x_hat))
    q_y = real(ifft(q_y_hat))
    
    jacobian = u_vel .* q_x .+ v_vel .* q_y
    jacobian_hat = fft(jacobian)
    
    # 3. Beta term: β * v
    # v is v_vel. 
    beta_term_hat = β .* v_vel_hat
    
    # 4. Dissipation: -r*∇^2ψ (Bottom friction) - ν*∇^4ψ (Hyperviscosity)
    # -r*(-q) = r*q
    # Actually standard form: q_t = -J(ψ, q + βy) - Dissipation + Forcing
    # Let's use: du = -J(ψ, q) - β*ψ_x - Dissipation
    
    # Explicit linear parts
    # Dissipation: -ν * (k^2)^p * q_hat
    dissipation = -ν .* (K2.^2) .* u
    
    @. du = -jacobian_hat - beta_term_hat + dissipation
end

function run_qg()
    N = 32
    L = 2π
    β = 10.0 # Beta plane effect (Planetary rotation)
    r = 0.1  # Bottom friction
    ν = 1e-3 # Viscosity
    
    k = [0:N/2-1; -N/2:-1]
    Kx = [i for i in k, j in k]
    Ky = [j for i in k, j in k]
    K2 = Kx.^2 .+ Ky.^2
    
    # Initial Condition: Decaying turbulence
    q0 = randn(N, N)
    q0_hat = fft(q0)
    
    tspan = (0.0, 40.0)
    prob = ODEProblem(qg_barotropic!, q0_hat, tspan, (Kx, Ky, K2, β, r, ν))
    sol = solve(prob, Tsit5(), saveat=1.0)
    
    # Plot final Potential Vorticity
    q_final = real(ifft(sol.u[end]))
    heatmap(q_final, title="5. QG Model (Potential Vorticity)", c=:deep, aspect_ratio=1)
end

# =========================================================
# Main Execution & Plotting
# =========================================================
println("Simulating 5 Advanced Benchmarks... (This may take a moment)")

p1 = run_ks()
p2 = run_l96_multi()
p3 = run_cgl()
p4 = run_kolmogorov()
p5 = run_qg()

# Layout
l = @layout [a b; c d; e]
final_plot = plot(p1, p2, p3, p4, p5, layout=l, size=(1000, 1200))

display(final_plot)
savefig("advanced_rc_benchmarks.png")






using DifferentialEquations
using LinearAlgebra
using Plots
using FFTW
using Random
using SparseArrays
using Statistics

default(fontfamily="Dejavu Sans Mono", titlefont=font(10), guidefont=font(8), margin=3Plots.mm)

# =========================================================
# 1. SABRA Shell Model of Turbulence
# 설명: 에너지 폭포(Energy Cascade)를 모사. 복소수 변수 u_n 사용.
# 난이도: 간헐성(Intermittency)과 급격한 에너지 전달.
# =========================================================
function sabra_model!(du, u, p, t)
    N, ν, k0, δ = p
    # k_n = k0 * 2^n (Wave numbers)
    # λ is usually 2.
    
    # Boundary conditions: u[-1] = u[-2] = 0 (handled by index check)
    a = 1.0; b = -0.5; c = 0.5 # Standard SABRA parameters
    
    for n in 1:N
        kn = k0 * 2.0^(n-1)
        kn_prev = (n > 1) ? k0 * 2.0^(n-2) : 0.0
        kn_prev2 = (n > 2) ? k0 * 2.0^(n-3) : 0.0
        
        u_next = (n < N) ? u[n+1] : 0.0im
        u_next2 = (n < N-1) ? u[n+2] : 0.0im
        u_prev = (n > 1) ? u[n-1] : 0.0im
        u_prev2 = (n > 2) ? u[n-2] : 0.0im
        
        # Nonlinear term: i * (a * k_n+1 * u_n+1 * u_n+2* ... )
        # Using the standard simplified form:
        # term1 = k_n * u_{n+1}^* * u_{n+2}
        # term2 = k_{n-1} * u_{n-1}^* * u_{n+1}
        # term3 = k_{n-2} * u_{n-1} * u_{n-2}
        
        nonlinear = 1.0im * (
            kn * conj(u_next) * u_next2 +
            b * kn_prev * conj(u_prev) * u_next - 
            (1+b) * kn_prev2 * u_prev * u_prev2 # Note: Different papers use slightly different (1+b) or c terms
        )
        
        # Dissipation + Forcing (only at low shells)
        forcing = (n == 1 || n == 2) ? (1.0 + 1.0im) : 0.0im
        
        du[n] = nonlinear - ν * (kn^2) * u[n] + forcing
    end
end

function run_sabra(regime)
    N = 20 # Number of shells
    k0 = 1.0
    tspan = (0.0, 100.0)
    u0 = (rand(N) .+ im.*rand(N)) .* 0.1
    
    if regime == :chaos
        ν = 1e-4 # Low viscosity -> Developed Turbulence
        title_str = "1A. SABRA: Fully Turbulent"
    else
        ν = 1e-2 # High viscosity -> Laminar/Decaying
        title_str = "1B. SABRA: Laminar/Decay"
    end
    
    prob = ODEProblem(sabra_model!, u0, tspan, (N, ν, k0, 0.0))
    sol = solve(prob, Tsit5(), saveat=0.1)
    
    # Plot log(|u_n|)
    data = reduce(hcat, [log10.(abs.(sol.u[i]) .+ 1e-9) for i in 1:length(sol.t)])'
    heatmap(1:N, sol.t, data, title=title_str, xlabel="Shell Index", ylabel="Time", c=:viridis)
end

# =========================================================
# 2. Barkley Model (2D Excitable Media)
# 설명: 심장 부정맥 모델. 나선파(Spiral Wave) 생성 및 붕괴.
# 난이도: 2D 반응-확산, 빠른 변수 변화.
# =========================================================
function barkley_step!(u, v, p, dt)
    # Explicit Euler with 5-point Laplacian for speed
    ϵ, a, b, D = p
    N = size(u, 1)
    
    # Laplacian function
    function laplacian(M)
        M_up = circshift(M, (-1, 0)); M_down = circshift(M, (1, 0))
        M_left = circshift(M, (0, -1)); M_right = circshift(M, (0, 1))
        return M_up .+ M_down .+ M_left .+ M_right .- 4.0 .* M
    end
    
    lu = laplacian(u)
    lv = laplacian(v)
    
    # Reaction terms
    # f(u, v) = 1/e * u * (1-u) * (u - (v+b)/a)
    # g(u, v) = u - v
    
    threshold = (v .+ b) ./ a
    reaction_u = (1/ϵ) .* u .* (1.0 .- u) .* (u .- threshold)
    reaction_v = u .- v
    
    @. u += dt * (reaction_u + D * lu)
    @. v += dt * (reaction_v + D * lv)
end

function run_barkley(regime)
    N = 80
    u = zeros(N, N)
    v = zeros(N, N)
    
    # Initial Spiral Seed (Cross gradient)
    for i in 1:N, j in 1:N
        if i > N/2; u[i,j] = 1.0; end
        if j > N/2; v[i,j] = 0.5; end
    end
    
    dt = 0.02
    steps = 2000
    
    if regime == :spiral
        # Stable Spiral
        ϵ = 0.02; a = 0.75; b = 0.01; D = 0.5 
        title_str = "2A. Barkley: Stable Spiral"
    else
        # Spiral Breakup (Chaos)
        # Reducing a makes the excitation threshold lower/faster
        ϵ = 0.02; a = 0.55; b = 0.05; D = 0.5
        title_str = "2B. Barkley: Spiral Chaos"
    end
    
    for t in 1:steps
        barkley_step!(u, v, (ϵ, a, b, D), dt)
    end
    
    heatmap(u, title=title_str, c=:inferno, aspect_ratio=1, axis=false)
end

# =========================================================
# 3. Spatially Coupled Mackey-Glass
# 설명: 지연 미분 방정식(DDE)의 격자 결합. Hyper-chaos.
# 구현: Ring Buffer를 이용한 수동 지연 처리 (효율성 위함)
# =========================================================
function run_mackey_glass_lattice(regime)
    N = 50 # Space
    steps = 1000
    dt = 0.1
    
    # Parameters
    beta = 2.0; gamma = 1.0; n = 10.0
    
    if regime == :periodic
        tau = 10.0 # Short delay -> Periodic/Stable
        D = 0.01   # Weak coupling
        title_str = "3A. MG Lattice: Periodic/Sync"
    else
        tau = 30.0 # Long delay -> Chaos
        D = 0.1    # Strong coupling -> Spatiotemporal Chaos
        title_str = "3B. MG Lattice: Hyperchaos"
    end
    
    tau_steps = Int(round(tau / dt))
    history = rand(N, tau_steps + 1) .* 0.5 .+ 0.5
    current_idx = 1
    
    data = zeros(steps, N)
    x = history[:, end]
    
    for t in 1:steps
        # Retrieve delayed state
        delayed_idx = mod1(current_idx - tau_steps, tau_steps + 1)
        x_tau = history[:, delayed_idx]
        
        # Calculate Laplacian (Coupling)
        lap = zeros(N)
        for i in 1:N
            im1 = mod1(i-1, N); ip1 = mod1(i+1, N)
            lap[i] = x[ip1] + x[im1] - 2*x[i]
        end
        
        # Euler Step: dx/dt = -gamma*x + beta*x_tau / (1 + x_tau^n) + D*lap
        dx = @. -gamma * x + beta * x_tau / (1.0 + x_tau^n) + D * lap
        x += dx * dt
        
        # Update History
        current_idx = mod1(current_idx + 1, tau_steps + 1)
        history[:, current_idx] = x
        data[t, :] = x
    end
    
    heatmap(data, title=title_str, c=:plasma, xlabel="Node", ylabel="Time")
end

# =========================================================
# 4. Rayleigh-Bénard Convection (Lorenz Lattice)
# 설명: 로렌츠 시스템을 공간적으로 결합하여 유체의 열 대류 모사.
# =========================================================
function lorenz_lattice!(du, u, p, t)
    # u structure: [x1, y1, z1, x2, y2, z2, ...]
    N, σ, ρ, β, κ = p
    
    for i in 1:N
        xi, yi, zi = u[3i-2], u[3i-1], u[3i]
        
        # Diffusive coupling on x (velocity)
        im1 = (i == 1) ? N : i-1
        ip1 = (i == N) ? 1 : i+1
        xim1, xip1 = u[3im1-2], u[3ip1-2]
        
        coupling = κ * (xip1 + xim1 - 2*xi)
        
        du[3i-2] = σ * (yi - xi) + coupling
        du[3i-1] = ρ * xi - yi - xi * zi
        du[3i]   = xi * yi - β * zi
    end
end

function run_lorenz_lattice(regime)
    N = 30
    σ = 10.0; β = 8/3
    
    if regime == :pattern
        ρ = 20.0 # Below standard chaos threshold (28) -> Stable fixed points or patterns
        κ = 5.0  # Strong coupling synchronizes neighbors
        title_str = "4A. Lorenz Lattice: Pattern"
    else
        ρ = 60.0 # High Rayleigh number -> Turbulence
        κ = 5.0  # Weak coupling allows local chaos
        title_str = "4B. Lorenz Lattice: Turbulence"
    end
    
    u0 = randn(3N)
    prob = ODEProblem(lorenz_lattice!, u0, (0.0, 40.0), (N, σ, ρ, β, κ))
    sol = solve(prob, Tsit5(), saveat=0.1)
    
    # Extract x variable for visualization
    data = reduce(hcat, [sol.u[i][1:3:end] for i in 1:length(sol.t)])'
    heatmap(data, title=title_str, c=:ice, xlabel="Lattice Index", ylabel="Time")
end

# =========================================================
# 5. Damped Nonlinear Schrödinger (DNLS)
# 설명: 광학 솔리톤 및 로그 파(Rogue Waves).
# =========================================================
function dnls_spectral!(du, u, p, t)
    # i ψ_t = -1/2 ψ_xx - |ψ|^2 ψ - i*gamma*ψ + Driving
    # Fourier: ψ_t = -i(-1/2 k^2)ψ - i*FFT(|ψ|^2 ψ) - gamma*ψ + ...
    k, γ, driving_amp = p
    
    ψ_real = ifft(u)
    nonlinear = abs2.(ψ_real) .* ψ_real
    nonlinear_hat = fft(nonlinear)
    
    # Driving (Homogeneous)
    driving_hat = zeros(ComplexF64, length(u))
    driving_hat[1] = driving_amp * length(u) # Only at k=0
    
    # Linear: -i * (-0.5 * k^2) = 0.5i * k^2
    # Damping: -γ
    linear = (0.5im .* k.^2 .- γ)
    
    @. du = linear * u + 1.0im * nonlinear_hat + driving_hat
end

function run_dnls(regime)
    N = 64
    L = 2π * 2
    dx = L/N
    k = [0:N/2-1; -N/2:-1] * (2π/L)
    
    if regime == :soliton
        # Stable Breather / Soliton regime
        γ = 0.1
        driving = 0.2
        title_str = "5A. DNLS: Stable Soliton"
        # Initial: Perturbed plane wave
        u0_real = 0.5 .* ones(ComplexF64, N) 
        u0_real[N÷2] += 1.0 # Spike
    else
        # Chaos / Rogue Wave regime
        γ = 0.05
        driving = 0.8 # High driving induces chaos
        title_str = "5B. DNLS: Rogue Chaos"
        u0_real = 0.5 .* ones(ComplexF64, N) .* (1 .+ 0.1*rand(N))
    end
    
    u0_hat = fft(u0_real)
    tspan = (0.0, 100.0)
    
    prob = ODEProblem(dnls_spectral!, u0_hat, tspan, (k, γ, driving))
    sol = solve(prob, Tsit5(), saveat=0.5)
    
    data = reduce(hcat, [abs.(ifft(sol.u[i])) for i in 1:length(sol.t)])'
    heatmap(data, title=title_str, c=:deep, xlabel="Space", ylabel="Time")
end

# =========================================================
# Main Execution & Plotting
# =========================================================
println("Generating 5 High-Complexity Benchmarks with State Comparisons...")

# 1. SABRA
p1a = run_sabra(:chaos)
p1b = run_sabra(:laminar)

# 2. Barkley
p2a = run_barkley(:spiral)
p2b = run_barkley(:chaos)

# 3. Mackey-Glass Lattice
p3a = run_mackey_glass_lattice(:periodic)
p3b = run_mackey_glass_lattice(:chaos)

# 4. Lorenz Lattice
p4a = run_lorenz_lattice(:pattern)
p4b = run_lorenz_lattice(:turbulence)

# 5. DNLS
p5a = run_dnls(:soliton)
p5b = run_dnls(:chaos)

# Combine all into a dashboard
l = @layout [a b; c d; e f; g h; i j]
final_plot = plot(p1a, p1b, p2a, p2b, p3a, p3b, p4a, p4b, p5a, p5b, 
                  layout=l, size=(1200, 2000), margin=5Plots.mm)

display(final_plot)
savefig("ultra_complex_benchmarks.png")





using DifferentialEquations
using Plots
using LinearAlgebra
using Statistics

default(fontfamily="Dejavu Sans Mono", titlefont=font(10), guidefont=font(8), margin=3Plots.mm)

# =========================================================
# 1. Lorenz-63 (The Butterfly)
# =========================================================
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    x, y, z = u
    du[1] = σ * (y - x)
    du[2] = x * (ρ - z) - y
    du[3] = x * y - β * z
end

function run_lorenz()
    u0 = [1.0, 0.0, 0.0]
    p = (10.0, 28.0, 8/3) # Standard Chaos Params
    tspan = (0.0, 50.0)
    prob = ODEProblem(lorenz!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=0.01)
    
    # 3D Plot
    plot(sol, vars=(1, 2, 3), title="1. Lorenz-63 Attractor", 
         xlabel="x", ylabel="y", zlabel="z", c=:plasma, lw=1.2, alpha=0.8)
end

# =========================================================
# 2. Rössler (The Folded Band)
# =========================================================
function rossler!(du, u, p, t)
    a, b, c = p
    x, y, z = u
    du[1] = -y - z
    du[2] = x + a * y
    du[3] = b + z * (x - c)
end

function run_rossler()
    u0 = [1.0, 1.0, 1.0]
    p = (0.2, 0.2, 5.7) # Standard Chaos Params
    tspan = (0.0, 100.0)
    prob = ODEProblem(rossler!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=0.05)
    
    plot(sol, vars=(1, 2, 3), title="2. Rössler Attractor", 
         xlabel="x", ylabel="y", zlabel="z", c=:viridis, lw=1.0)
end

# =========================================================
# 3. Mackey-Glass (Time Delay)
# Note: Manually implemented for portability (RK4 with History)
# =========================================================
function run_mackey_glass()
    steps = 2000
    dt = 0.1
    β = 0.2; γ = 0.1; n = 10.0; τ = 17.0 # Standard Chaos
    
    tau_steps = Int(round(τ / dt))
    history = zeros(tau_steps + 1) .+ 1.2 # Initial history
    x = 1.2
    
    data = Float64[]
    push!(data, x)
    
    for t in 1:steps
        # Get delayed value
        x_tau = history[1]
        
        # RK4 Integration step
        dx_dt = (x_val, x_del) -> β * x_del / (1 + x_del^n) - γ * x_val
        
        k1 = dx_dt(x, x_tau)
        # For simplicity in delay handling, we use Euler-like step or 
        # assume history is constant for fractional step. 
        # Here we use simple Euler for readability in snippet.
        x_next = x + k1 * dt
        
        # Update history (Ring buffer shift)
        popfirst!(history)
        push!(history, x_next)
        
        x = x_next
        push!(data, x)
    end
    
    # Plot Phase Space (x(t) vs x(t-tau))
    plot(data[1:end-tau_steps], data[tau_steps+1:end], 
         title="3. Mackey-Glass Phase Space", 
         xlabel="x(t)", ylabel="x(t-τ)", c=:magma, lw=1.5)
end

# =========================================================
# 4. Duffing Oscillator (Driven)
# =========================================================
function duffing!(du, u, p, t)
    δ, α, β, γ, ω = p
    x, v = u
    du[1] = v
    du[2] = -δ * v - α * x - β * x^3 + γ * cos(ω * t)
end

function run_duffing()
    u0 = [0.1, 0.1]
    # Chaotic parameters
    p = (0.3, -1.0, 1.0, 0.5, 1.2) 
    tspan = (0.0, 100.0)
    prob = ODEProblem(duffing!, u0, tspan, p)
    sol = solve(prob, Tsit5(), saveat=0.05)
    
    plot(sol, vars=(1, 2), title="4. Duffing Phase Space", 
         xlabel="Position (x)", ylabel="Velocity (v)", c=:inferno, lw=0.8)
end

# =========================================================
# 5. Kuramoto Model (Synchronization)
# =========================================================
function kuramoto!(du, u, p, t)
    K, ω = p # K: Coupling, ω: Natural frequencies
    N = length(u)
    
    for i in 1:N
        coupling = 0.0
        for j in 1:N
            coupling += sin(u[j] - u[i])
        end
        du[i] = ω[i] + (K / N) * coupling
    end
end

function run_kuramoto()
    N = 20
    K = 2.0 # Coupling strength > Kc implies sync
    ω = randn(N) # Random natural frequencies
    u0 = 2π * rand(N)
    
    tspan = (0.0, 20.0)
    prob = ODEProblem(kuramoto!, u0, tspan, (K, ω))
    sol = solve(prob, Tsit5(), saveat=0.1)
    
    # Heatmap of phases over time
    data = reduce(hcat, sol.u)'
    # Map to -pi to pi for visualization
    data_mod = mod2pi.(data) .- π
    
    heatmap(sol.t, 1:N, data_mod', title="5. Kuramoto Sync", 
            xlabel="Time", ylabel="Oscillator Index", c=:twilight)
end

# =========================================================
# Execution
# =========================================================
p1 = run_lorenz()
p2 = run_rossler()
p3 = run_mackey_glass()
p4 = run_duffing()
p5 = run_kuramoto()

l = @layout [a b; c d; e]
final_plot = plot(p1, p2, p3, p4, p5, layout=l, size=(1000, 1200))
display(final_plot)
savefig(final_plot, "classic_rc_benchmarks.png")