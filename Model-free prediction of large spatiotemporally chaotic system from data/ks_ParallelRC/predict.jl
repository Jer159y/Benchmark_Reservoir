using Distributed

function predict_parallel(w, w_out, x, w_in, pl, chunk_size, frontWkrIdx, rearWkrIdx, N, locality, channels)
    # channels: Vector of RemoteChannel, 워커 간 통신용
    # my_idx: 현재 워커의 인덱스 (1부터 시작)
    
    prediction = zeros(Float64, Int(chunk_size), Int(pl))
    curr_x = copy(x)
    
    my_idx = myid() # 현재 프로세스 ID (topology에 따라 조정 필요할 수 있음)
    
    # 통신 채널 매핑 (topology가 원형이라고 가정)
    # channels[i]는 i번 워커가 '받는' 채널이라고 가정하거나, 
    # 별도의 send/recv 채널 구조체를 만들어야 함.
    # 여기서는 간단히 channels 딕셔너리를 사용한다고 가정: channels[(from, to)]
    
    for i = 1:pl
        # Feature augmentation
        x_aug = copy(curr_x)
        x_aug[2:2:end] .= x_aug[2:2:end] .^ 2
        
        # 출력 계산
        out = w_out * x_aug
        
        # --- 병렬 통신 (labSendReceive 대체) ---
        
        # 1. 보낼 데이터 준비
        data_to_rear = out[end-locality+1:end]
        data_to_front = out[1:locality]
        
        # 2. 비동기 전송 (put!)
        # 내 뒤쪽 워커(rearWkrIdx)에게 data_to_rear를 보냄
        # 내 앞쪽 워커(frontWkrIdx)에게 data_to_front를 보냄
        # 키 형식: (보내는사람, 받는사람)
        @async put!(channels[(my_idx, rearWkrIdx)], data_to_rear)
        @async put!(channels[(my_idx, frontWkrIdx)], data_to_front)
        
        # 3. 데이터 수신 (take!)
        # 앞쪽 워커가 나에게 보낸 데이터 (front_out) -> 즉, 나는 rearWkrIdx 입장에서 front임
        # MATLAB 로직: rear_out = labSendReceive(frontWkrIdx, rearWkrIdx, data_to_rear)
        # -> "front에게 보내고, front로부터 받는다" (데이터의 내용은 rear쪽 오버랩)
        
        # Julia로 명시적 해석:
        # 내가 필요한 것:
        # - rear_out: 내 앞쪽(Front) 워커의 뒷부분 데이터
        # - front_out: 내 뒤쪽(Rear) 워커의 앞부분 데이터
        
        rear_out = take!(channels[(frontWkrIdx, my_idx)])
        front_out = take!(channels[(rearWkrIdx, my_idx)])
        
        # --- 피드백 벡터 구성 ---
        feedback = vcat(rear_out, out, front_out)
        
        # 상태 업데이트
        curr_x = tanh.(w * curr_x .+ w_in * feedback)
        
        prediction[:, i] = out
    end
    
    return prediction
end