using SparseArrays
using Random
using LinearAlgebra

# 의존성 함수들 (같은 모듈이나 include로 로드되어야 함)
# include("generate_reservoir.jl")
# include("reservoir_layer.jl")
# include("fit.jl")

function train_reservoir(resparams, data, labindex, jobid, locality, chunk_size)
    num_inputs = size(data, 1)
    
    # Reservoir 행렬 A 생성
    A = generate_reservoir(Int(resparams[:N]), resparams[:radius], Int(resparams[:degree]), labindex, jobid)
    
    # 입력 가중치 win 생성
    N = Int(resparams[:N])
    q = div(N, num_inputs)
    win = zeros(Float64, N, num_inputs)
    
    # MATLAB: rng(i) 로직 재현
    # 주의: Julia와 MATLAB의 난수 생성 알고리즘이 다르므로 값은 다르지만,
    # 시드 고정 로직은 동일하게 구현합니다.
    for i = 1:num_inputs
        Random.seed!(i)
        ip = -1.0 .+ 2.0 .* rand(q)
        
        start_idx = (i-1)*q + 1
        end_idx = i*q
        win[start_idx:end_idx, i] = ip
    end
    
    # 상태 수집 (reservoir_layer)
    states = reservoir_layer(A, win, data, resparams)
    
    # Feature transformation: 짝수 행 제곱
    # MATLAB: states(2:2:resparams.N,:) = states(2:2:resparams.N,:).^2;
    states[2:2:end, :] .= states[2:2:end, :] .^ 2
    
    # 학습 데이터 슬라이싱 (Target data)
    # MATLAB: data(locality+1:locality+chunk_size, ...)
    # 인덱스 주의: Julia도 1-based indexing이므로 그대로 사용 가능하나 정수형 변환 필요
    target_start_row = Int(locality) + 1
    target_end_row = Int(locality) + Int(chunk_size)
    
    discard_len = Int(resparams[:discard_length])
    train_len = Int(resparams[:train_length])
    
    target_data = data[target_start_row:target_end_row, discard_len+1 : discard_len+train_len]
    
    # 학습 (fit)
    wout = fit(resparams, states, target_data)
    
    # 마지막 상태 반환
    x = states[:, end]
    
    return x, wout, A, win
end