function reservoir_layer(A, win, input, resparams)
    N = Int(resparams[:N])
    len = Int(resparams[:train_length])
    
    states = zeros(Float64, N, len)

    if !all(isfinite, input)
        error("Non-finite values detected in reservoir input")
    end
    
    # 첫 번째 상태는 0으로 시작한다고 가정 (MATLAB 코드 흐름 따름)
    # MATLAB 코드에서는 loop가 1부터 train_length-1까지 돌며 i+1에 저장함.
    # 즉, states[:, 1]은 0으로 유지됨.
    
    for i = 1:len-1
        # tanh.(...) 브로드캐스팅 사용
        states[:, i+1] = tanh.(A * states[:, i] .+ win * input[:, i])
    end
    
    return states
end