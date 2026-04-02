using LinearAlgebra
using SparseArrays

function train(params, states, data)
    beta = params[:beta]
    N = Int(params[:N])
    
    idenmat = beta * I(N) # 단위 행렬
    
    # 짝수 인덱스 제곱 (Feature transformation)
    # Julia 인덱싱: 2:2:end
    states_aug = copy(states)
    states_aug[2:2:end, :] .= states_aug[2:2:end, :] .^ 2
    
    if size(states_aug, 1) != N
        error("states row count (" * string(size(states_aug, 1)) * ") != N (" * string(N) * ")")
    end
    if !all(isfinite, states_aug) || !all(isfinite, data)
        error("Non-finite values detected in training data or states")
    end

    # Ridge Regression: W_out = Y * X^T * (X * X^T + beta*I)^-1
    # SVD 기반 pinv 대신 Cholesky로 안정적인 선형해 풀이
    M = states_aug * states_aug' .+ idenmat
    w_out = (data * states_aug') / cholesky(Symmetric(M))
    
    return w_out
end