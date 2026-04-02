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