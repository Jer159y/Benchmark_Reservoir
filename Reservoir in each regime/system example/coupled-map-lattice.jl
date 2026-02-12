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