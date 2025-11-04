using LinearAlgebra, SparseArrays, Printf

mutable struct ReservoirTanh
    # Matrices
    A::SparseMatrixCSC{Float64, Int}
    B::Matrix{Float64}
    C::Matrix{Float64}
    R::Matrix{Float64}  # R은 predict_x에서 계산됨
    
    # States and fixed points
    r::Vector{Float64}
    rs::Vector{Float64}
    xs::Vector{Float64}
    cs::Vector{Float64}
    d::Vector{Float64}
    
    # Time
    delT::Float64
    gam::Float64

    # 생성자
    function ReservoirTanh(A, B, C, rs, xs, cs, delT, gam)
        N = size(A, 1)
        d = atanh.(rs) .- A*rs .- B*xs .- C*cs
        r = zeros(N)
        R = zeros(size(A,1), N) # R은 나중에 채워짐
        new(A, B, C, R, r, rs, xs, cs, d, delT, gam)
    end
end

# ODEs (driven)
function del_r(o::ReservoirTanh, r, x, c)
    return o.gam .* (-r .+ tanh.(o.A*r .+ o.B*x .+ o.C*c .+ o.d))
end

# ODEs (feedback)
function del_r_x(o::ReservoirTanh, r, c)
    return o.gam .* (-r .+ tanh.(o.R*r .+ o.C*c .+ o.d))
end

# RK4 integrator (driven)
function propagate!(o::ReservoirTanh, x, c)
    if ndims(x) == 2
        x = reshape(x, size(x, 1), 1, size(x, 2))
    end
    if ndims(c) == 2
        c = reshape(c, size(c, 1), 1, size(c, 2))
    end
    # x, c는 [:, 1, 1] ... [:, 1, 4] 형태의 3D 배열로 가정
    k1 = o.delT * del_r(o, o.r,        x[:, 1, 1], c[:, 1, 1])
    k2 = o.delT * del_r(o, o.r .+ k1/2, x[:, 1, 2], c[:, 1, 2])
    k3 = o.delT * del_r(o, o.r .+ k2/2, x[:, 1, 3], c[:, 1, 3])
    k4 = o.delT * del_r(o, o.r .+ k3,   x[:, 1, 4], c[:, 1, 4])
    o.r = o.r .+ (k1 .+ 2*k2 .+ 2*k3 .+ k4) / 6
end

# RK4 integrator (feedback)
function propagate_x!(o::ReservoirTanh, c)
    if ndims(c) == 2
        c = reshape(c, size(c, 1), 1, size(c, 2))
    end
    k1 = o.delT * del_r_x(o, o.r,        c[:, 1, 1])
    k2 = o.delT * del_r_x(o, o.r .+ k1/2, c[:, 1, 2])
    k3 = o.delT * del_r_x(o, o.r .+ k2/2, c[:, 1, 3])
    k4 = o.delT * del_r_x(o, o.r .+ k3,   c[:, 1, 4])
    o.r = o.r .+ (k1 .+ 2*k2 .+ 2*k3 .+ k4) / 6
end

# Training
function train!(o::ReservoirTanh, x, c)
    nx = size(x, 2)
    D = zeros(size(o.A, 1), nx)
    D[:, 1] = o.r
    
    @printf("%s\n", "."^100)
    nInd = 0.0
    
    for i = 2:nx
        if i > nInd * nx
            @printf("=")
            nInd += 0.01
        end
        # x, c의 인덱싱 주의: MATLAB (:, i-1, :) -> Julia (:, i-1, :)
        propagate!(o, x[:, i-1, :], c[:, i-1, :])
        D[:, i] = o.r
    end
    
    @printf("\n")
    return D
end

# Prediction
function predict_x!(o::ReservoirTanh, c, W)
    nc = size(c, 2)
    o.R = o.A + o.B * W  # Feedback
    D = zeros(size(o.R, 1), nc)
    D[:, 1] = o.r
    
    @printf("%s\n", "."^100)
    nInd = 0.0
    
    for i = 2:nc
        if i > nInd * nc
            @printf("=")
            nInd += 0.01
        end
        propagate_x!(o, c[:, i-1, :])
        D[:, i] = o.r
    end
    
    @printf("\n")
    return D
end