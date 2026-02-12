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