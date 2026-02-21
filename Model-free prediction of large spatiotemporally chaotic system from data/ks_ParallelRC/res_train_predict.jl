function res_train_predict(train_in, test_in, resparams, jobid, locality, chunk_size, pred_marker_array, sync_length, channels, front_pid, rear_pid)
    
    # 현재 워커의 labindex(논리적 인덱스)는 함수 내부에서 필요하다면 계산해서 써야 하지만
    # train_reservoir는 주로 로컬 연산이므로 labindex는 시드 생성용으로 쓰임.
    # 여기서는 간단히 jobid를 활용하거나 Random 시드를 위해 myid() 사용.
    
    # 1. 학습
    x, w_out, w, w_in = train_reservoir(resparams, train_in, myid(), jobid, locality, chunk_size)
    
    num_preds = length(pred_marker_array)
    pred_len = Int(resparams[:predict_length])
    
    pred_collect = zeros(Float64, Int(chunk_size), num_preds * pred_len)
    
    for pred_iter = 1:num_preds
        prediction_marker = Int(pred_marker_array[pred_iter])
        
        # 2. 동기화 (Washing out)
        x_sync = synchronize(w, x, w_in, test_in, prediction_marker, sync_length)
        
        # 3. 예측 (병렬 통신 포함)
        prediction = predict_parallel(
            w, w_out, x_sync, w_in, pred_len, chunk_size, 
            front_pid, rear_pid, Int(resparams[:N]), locality, channels
        )
        
        # 결과 저장
        col_start = (pred_iter-1)*pred_len + 1
        col_end = pred_iter*pred_len
        pred_collect[:, col_start:col_end] = prediction
    end
    
    return pred_collect
end