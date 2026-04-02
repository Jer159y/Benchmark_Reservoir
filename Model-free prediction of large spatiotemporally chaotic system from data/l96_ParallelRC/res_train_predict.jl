function res_train_predict(in_data, test_in, resparams, jobid, locality, chunk_size, pred_marker_array, channels, front_pid, rear_pid)
    
    # 1. Train
    # labindex는 현재 프로세스의 논리적 인덱스가 필요하나, 여기선 간단히 1로 가정하거나
    # 상위에서 넘겨받아야 함. 시드 생성용이므로 jobid와 함께 유니크하면 됨.
    current_labindex = myid() # Simplified
    
    x, w_out, w, w_in = train_reservoir(resparams, in_data, current_labindex, jobid, locality, chunk_size)
    
    sync_length = 32
    num_preds = length(pred_marker_array)
    pred_len = Int(resparams[:predict_length])
    
    pred_collect = zeros(Float64, Int(chunk_size), num_preds * pred_len)
    
    for pred_iter = 1:num_preds
        prediction_marker = Int(pred_marker_array[pred_iter])
        
        # 2. Synchronize
        x_sync = synchronize(w, x, w_in, test_in, prediction_marker, sync_length)
        
        # 3. Predict (Parallel)
        prediction = predict_parallel(w, w_out, x_sync, w_in, pred_len, chunk_size, 
                                      front_pid, rear_pid, Int(resparams[:N]), locality, channels)
        
        col_start = (pred_iter-1)*pred_len + 1
        col_end = pred_iter*pred_len
        pred_collect[:, col_start:col_end] = prediction
    end
    
    return pred_collect
end