function predict(A, win, resparams, x_init, w_out)
    predict_length = Int(resparams[:predict_length])
    num_inputs = Int(resparams[:num_inputs])
    N = Int(resparams[:N])
    
    output = zeros(Float64, num_inputs, predict_length)
    
    x = copy(x_init)
    
    for i = 1:predict_length
        # Feature transformation (짝수 인덱스 제곱)
        x_aug = copy(x)
        x_aug[2:2:N] .= x_aug[2:2:N] .^ 2
        
        # 출력 계산
        out = w_out * x_aug
        output[:, i] = out
        
        # Reservoir 상태 업데이트 (Closed-loop: 출력이 다시 입력으로 들어감)
        x = tanh.(A * x .+ win * out)
    end
    
    return output, x
end