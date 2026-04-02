using Random

# 앞서 정의한 함수들이 필요하므로 include 혹은 같은 파일 내에 있어야 함
# include("generate_reservoir.jl")
# include("reservoir_layer.jl")
# include("train.jl")

function train_reservoir(resparams, data)
    if !all(isfinite, data)
        error("Non-finite values detected in training data")
    end

    # Reservoir 행렬 생성
    A = generate_reservoir(Int(resparams[:N]), resparams[:radius], Int(resparams[:degree]))
    
    num_inputs = Int(resparams[:num_inputs])
    N = Int(resparams[:N])
    q = div(N, num_inputs)
    
    win = zeros(Float64, N, num_inputs)
    
    # 입력 가중치 생성 (MATLAB의 rng(i) 재현)
    # Julia의 RNG 동작이 MATLAB과 다르므로 숫자는 정확히 같지 않지만 로직은 유지
    for i = 1:num_inputs
        Random.seed!(i) 
        ip = resparams[:sigma] .* (-1 .+ 2 .* rand(q))
        
        start_idx = (i-1)*q + 1
        end_idx = i*q
        win[start_idx:end_idx, i] = ip
    end
    if !all(isfinite, win)
        error("Non-finite values detected in input weights (win)")
    end
    if !all(isfinite, A)
        error("Non-finite values detected in reservoir matrix (A)")
    end
    
    # 상태 수집
    states = reservoir_layer(A, win, data, resparams)
    if !all(isfinite, states)
        error("Non-finite values detected in reservoir states")
    end
    
    # 학습
    wout = train(resparams, states, data)
    
    # 마지막 상태 반환
    x = states[:, end]
    
    return x, wout, A, win
end