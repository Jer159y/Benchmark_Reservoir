using Distributed

function predict_parallel(w, w_out, x, w_in, pl, chunk_size, frontWkrIdx, rearWkrIdx, N, locality, channels)
    # channels: Dict{Tuple{Int,Int}, RemoteChannel}
    # pl: predict_length
    
    prediction = zeros(Float64, Int(chunk_size), Int(pl))
    curr_x = copy(x)
    my_pid = myid()
    
    for i = 1:pl
        # Feature Augmentation
        x_aug = copy(curr_x)
        x_aug[2:2:end] .= x_aug[2:2:end] .^ 2
        
        # Output calculation
        out = w_out * x_aug
        
        # --- Communication (Replacing labSendReceive) ---
        # MATLAB Logic:
        # rear_out = labSendReceive(frontWkrIdx, rearWkrIdx, out(end-locality+1:end));
        # -> Send TAIL to FRONT, Receive from REAR (which is rear's HEAD logic? No, wait)
        
        # Let's align with MATLAB exactly:
        # "Send to target, Receive from source"
        
        # 1. Prepare data
        data_to_front = out[end-locality+1:end] # Tail sent to front worker
        data_to_rear  = out[1:locality]          # Head sent to rear worker
        
        # 2. Async Send
        # I send my tail to front worker
        @async put!(channels[(my_pid, frontWkrIdx)], data_to_front)
        # I send my head to rear worker
        @async put!(channels[(my_pid, rearWkrIdx)], data_to_rear)
        
        # 3. Receive (Blocking)
        # I receive from rear worker (who sent me their tail)
        # rear_out corresponds to the data coming from the 'rear' direction
        rear_out = take!(channels[(rearWkrIdx, my_pid)])
        
        # I receive from front worker (who sent me their head)
        front_out = take!(channels[(frontWkrIdx, my_pid)])
        
        # 4. Feedback construction
        feedback = vcat(rear_out, out, front_out)
        
        # State Update
        curr_x = tanh.(w * curr_x .+ w_in * feedback)
        
        prediction[:, i] = out
    end
    
    return prediction
end