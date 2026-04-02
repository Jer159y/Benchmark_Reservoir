function synchronize(W, x, w_in, data, prediction_marker, sync_length)
    # x: 현재 상태 벡터
    x_new = copy(x)
    
    for i = 1:sync_length
        # data[:, prediction_marker + i]는 해당 시점의 입력 벡터
        # MATLAB: tanh(W*x + w_in*data)
        # Julia: 브로드캐스팅(.) 사용
        input_vec = data[:, prediction_marker + i]
        x_new = tanh.(W * x_new .+ w_in * input_vec)
    end
    
    return x_new
end