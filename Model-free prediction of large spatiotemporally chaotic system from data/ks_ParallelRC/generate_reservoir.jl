# generate_reservoir.jl
using SparseArrays
using Arpack
using Random

function generate_reservoir(size::Int, radius::Float64, degree::Int, labindex, jobid)
    # 시드 설정에 labindex와 jobid 사용
    Random.seed!(labindex + jobid)
    
    sparsity = degree / size
    A = sprand(size, size, sparsity)
    
    # 스펙트럼 반경 조정
    vals, _ = eigs(A; nev=1, which=:LM, ritzvec=false)
    e = maximum(abs.(vals))
    
    A = (A ./ e) .* radius
    return A
end

# fit.jl
using LinearAlgebra
using SparseArrays

function fit(params, states, data)
    beta = params[:beta]
    N = Int(params[:N])
    
    # idenmat = beta * I
    idenmat = beta * I(N)
    
    # MATLAB: w_out = data * states' * pinv(states * states' + idenmat)
    # Julia pinv는 SVD 기반이라 느릴 수 있음. '\' 연산자(Backslash)가 더 빠르고 안정적일 수 있으나
    # 정확한 변환을 위해 수식대로 구현함.
    
    # data: (Output Dim x Time)
    # states: (Reservoir Dim x Time)
    
    w_out = data * states' * pinv(states * states' + idenmat)
    return w_out
end

# reservoir_layer.jl
# (Batch 1 버전과 동일, 입력 데이터 인덱싱 주의)
function reservoir_layer(A, win, input, resparams)
    N = Int(resparams[:N])
    train_len = Int(resparams[:train_length])
    discard_len = Int(resparams[:discard_length])
    
    states = zeros(Float64, N, train_len)
    x = zeros(Float64, N)
    
    # Transient discard
    for i = 1:discard_len
        x = tanh.(A * x .+ win * input[:, i])
    end
    
    states[:, 1] = x
    
    # State collection
    # MATLAB 코드: for i = 1:train_length-1 ... input(:, discard + i)
    # states(:, i+1) = ...
    # 즉, states 1열은 discard 직후 상태.
    
    for i = 1:train_len-1
        # input 인덱스: discard_len + i
        # 다음 상태 계산
        x_next = tanh.(A * states[:, i] .+ win * input[:, discard_len + i])
        states[:, i+1] = x_next
    end
    
    return states
end