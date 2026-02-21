# 의존성: l96_helpers.jl 함수들
function train_reservoir(resparams, data, labindex, jobid, locality, chunk_size)
    num_inputs = size(data, 1) # 행이 입력 차원이라고 가정 (MATLAB 코드 기반)
    
    # Reservoir 생성
    A = generate_reservoir(Int(resparams[:N]), resparams[:radius], Int(resparams[:degree]), labindex, jobid)
    
    N = Int(resparams[:N])
    q = div(N, num_inputs)
    win = zeros(Float64, N, num_inputs)
    
    # Input Weight 생성 (rng(i) 로직 재현)
    for i = 1:num_inputs
        Random.seed!(i)
        ip = -1.0 .+ 2.0 .* rand(q)
        win[(i-1)*q+1 : i*q, i] = ip
    end
    
    # 상태 수집
    states = reservoir_layer(A, win, data, resparams)
    
    # Feature Augmentation (짝수 행 제곱)
    states[2:2:end, :] .= states[2:2:end, :] .^ 2
    
    # Target Data Slicing
    # MATLAB: data(locality+1:locality+chunk_size, discard+1 : discard+train)
    discard_len = Int(resparams[:discard_length])
    train_len = Int(resparams[:train_length])
    
    # 인덱스 주의: Julia 1-based
    target_rows = (Int(locality)+1) : (Int(locality)+Int(chunk_size))
    target_cols = (discard_len+1) : (discard_len+train_len)
    
    target_data = data[target_rows, target_cols]
    
    # 학습 (Ridge Regression)
    wout = fit(resparams, states, target_data)
    
    # 마지막 상태
    x_final = states[:, end]
    
    return x_final, wout, A, win
end