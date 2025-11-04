using LinearAlgebra, SparseArrays, Printf

mutable struct ReservoirWC
    # Matrices
    A::SparseMatrixCSC{Float64, Int}
    B::Matrix{Float64}
    C::Matrix{Float64}
    R::Matrix{Float64}
    W::Matrix{Float64} # W는 predict_x에서 설정됨
    re::Float64
    
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
    function ReservoirWC(A, B, C, rs, xs, cs, delT, gam)
        N = size(A, 1)
        re = 0.0
        # MATLAB: log(rs ./ (1 - (1+obj.re).*rs))
        # Julia: log.(rs ./ (1 .- (1+re).*rs)) (브로드캐스팅 주의)
        d = -A*rs .- B*xs .- C*cs .+ log.(rs ./ (1 .- (1+re).*rs))
        r = zeros(N)
        R = zeros(N, N)
        W = zeros(size(B, 2), N) # 임시 크기, predict_x에서 덮어씀
        
        new(A, B, C, R, W, re, r, rs, xs, cs, d, delT, gam)
    end
end

# ODEs (driven)
function del_r(o::ReservoirWC, r, x, c)
    # 브로드캐스팅 주의: (1 .- o.re .* r) ./ (1 .+ exp.(...))
    dr = o.gam .* (-r .+ (1 .- o.re .* r) ./ (1 .+ exp.(-o.A*r .- o.B*x .- o.C*c .- o.d)))
    return dr
end

# ODEs (feedback)
function del_r_x(o::ReservoirWC, r, c)
    dr = o.gam .* (-r .+ (1 .- o.re .* r) ./ (1 .+ exp.(-o.A*r .- o.B*(o.W*r) .- o.C*c .- o.d)))
    return dr
end

# RK4 integrators (ReservoirTanh와 동일한 구조)
# propagate!(o::ReservoirWC, x, c) ...
# propagate_x!(o::ReservoirWC, c) ...

# train! 및 predict_x! (ReservoirTanh와 동일한 구조)
# train!(o::ReservoirWC, x, c) ...
# function predict_x!(o::ReservoirWC, c, W) ...
#   o.R = o.A + o.B * W
#   o.W = W
# ...