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

# -----------------------------------------------------------
# Example: Solve and plot each dynamical system
# -----------------------------------------------------------
using Random
using LinearAlgebra
using DifferentialEquations
using Plots
gr()

function example_stuart_landau(; N=5, tspan=(0.0, 80.0), dt=0.05)
    Random.seed!(42)
    λ = 0.5 .+ 0.1 .* randn(N)
    ω = 1.0 .+ 0.2 .* randn(N)
    Id = Matrix{Float64}(I, N, N)
    K = 0.08 .* (ones(N, N) .- Id)
    u0 = 0.1 .* randn(ComplexF64, N)
    prob = ODEProblem(stuart_landau!, u0, tspan, (N, λ, ω, K))
    sol = solve(prob, Tsit5(); saveat=dt)
    p = plot(title="Stuart-Landau: |z_i|", xlabel="t", ylabel="|z|", legend=:right)
    for i in 1:N
        plot!(p, sol.t, abs.([u[i] for u in sol.u]), label="i=$i")
    end
    return p
end

function example_HR_net(; N=5, tspan=(0.0, 200.0), dt=0.2)
    Random.seed!(43)
    I0 = 3.0 .+ 0.1 .* randn(N)
    Id = Matrix{Float64}(I, N, N)
    K = 0.02 .* (ones(N, N) .- Id)
    u0 = reduce(vcat, ([0.1, 0.0, 0.0] .+ 0.05 .* randn(3) for _ in 1:N))
    prob = ODEProblem(HR_net!, u0, tspan, (N, I0, K))
    sol = solve(prob, Tsit5(); saveat=dt)
    p = plot(title="Hindmarsh-Rose: x_1", xlabel="t", ylabel="x_1")
    plot!(p, sol.t, [u[1] for u in sol.u], label="x1")
    return p
end

function example_adaptive_kuramoto(; N=6, tspan=(0.0, 100.0), dt=0.1)
    Random.seed!(44)
    ω = 0.5 .+ 0.2 .* randn(N)
    ε = 0.2
    Id = Matrix{Float64}(I, N, N)
    A = ones(N, N) .- Id
    θ0 = 2π .* rand(N)
    κ0 = zeros(N, N)
    u0 = vcat(θ0, vec(κ0))
    prob = ODEProblem(adaptive_kuramoto!, u0, tspan, (N, ω, ε, A))
    sol = solve(prob, Tsit5(); saveat=dt)
    p = plot(title="Adaptive Kuramoto: θ_i", xlabel="t", ylabel="θ", legend=:right)
    for i in 1:N
        plot!(p, sol.t, [u[i] for u in sol.u], label="i=$i")
    end
    return p
end

function example_rossler_net(; N=4, tspan=(0.0, 200.0), dt=0.2)
    Random.seed!(45)
    a = 0.2 .+ 0.02 .* randn(N)
    b = 0.2 .+ 0.02 .* randn(N)
    c = 5.7 .+ 0.1 .* randn(N)
    Id = Matrix{Float64}(I, N, N)
    K = 0.05 .* (ones(N, N) .- Id)
    u0 = reduce(vcat, ([0.1, 0.0, 0.0] .+ 0.05 .* randn(3) for _ in 1:N))
    prob = ODEProblem(rossler_net!, u0, tspan, (N, a, b, c, K))
    sol = solve(prob, Tsit5(); saveat=dt)
    p = plot(title="Rössler: x_1", xlabel="t", ylabel="x_1")
    plot!(p, sol.t, [u[1] for u in sol.u], label="x1")
    return p
end

function example_cml(; N=50, steps=200)
    Random.seed!(46)
    r = 3.8 .+ 0.02 .* randn(N)
    ε = 0.15
    Id = Matrix{Float64}(I, N, N)
    A = (ones(N, N) .- Id) ./ (N-1)
    x = 0.2 .+ 0.1 .* rand(N)
    X = zeros(steps, N)
    for t in 1:steps
        x = cml_step!(x, r, ε, A)
        X[t, :] = x
    end
    p = heatmap(1:N, 1:steps, X, xlabel="node", ylabel="step", title="CML states")
    return p
end


p1 = example_stuart_landau()
p2 = example_HR_net()
p3 = example_adaptive_kuramoto()
p4 = example_rossler_net()
p5 = example_cml()
plt = plot(p1, p2, p3, p4, p5, layout=(3,2), size=(1000, 900))
display(plt)
