using Distributed
using MAT
using JLD2

# 프로세스 설정 (예: 워커 4개)
if nprocs() == 1
    addprocs(4) # 실제 환경에 맞춰 조정
end

@everywhere begin
    using SparseArrays, LinearAlgebra, Statistics, Random
    # 파일들을 include (경로는 실제 환경에 맞게 수정)
    include("l96_helpers.jl")
    include("train_reservoir.jl")
    include("predict.jl")
    include("res_train_predict.jl")
end

function main_parallel_benchmark(pool_size)
    # --- 데이터 로드 (경로 수정 필요) ---
    # m = matread("/path/to/train_input_sequence.mat")
    # train_seq = m["train_input_sequence"]
    
    # [데모용 데이터 생성]
    println("Generating dummy L96 data...")
    total_dim = 20 # 예: 차원 20
    train_seq = rand(80000, total_dim)
    test_seq = rand(20000, total_dim)
    full_pred_marker_array = [40000] # 테스트용 마커
    
    sigma = 0.1
    
    # --- 채널 토폴로지 구성 (Ring) ---
    workers_list = workers()[1:pool_size]
    channels = Dict{Tuple{Int,Int}, RemoteChannel}()
    
    for (i, pid) in enumerate(workers_list)
        # Circular topology indices
        next_idx = mod(i, pool_size) + 1
        prev_idx = mod(i - 2, pool_size) + 1
        
        front_pid = workers_list[next_idx]
        rear_pid  = workers_list[prev_idx]
        
        # Create channels (Buffer size > 0 important for non-blocking put!)
        channels[(pid, front_pid)] = RemoteChannel(()->Channel{Vector{Float64}}(10))
        channels[(pid, rear_pid)]  = RemoteChannel(()->Channel{Vector{Float64}}(10))
    end
    
    # --- 작업 분산 ---
    futures = []
    
    for (idx, pid) in enumerate(workers_list)
        f = @spawnat pid begin
            # Worker-local variable setup
            labindex = idx
            numlabs = pool_size
            jobid = 1
            
            # Data partitioning
            len, num_inputs = size(train_seq)
            chunk_size = div(num_inputs, numlabs)
            
            chunk_begin = chunk_size * (labindex - 1) + 1
            chunk_end = chunk_size * labindex
            locality = 2 # from benchmarking_pv.m
            
            # Calculate overlaps
            rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs)
            forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs)
            overlap_size = length(rear_overlap) + length(forward_overlap)
            
            # Construct Input Data (u)
            # MATLAB: u(:,1:locality) = ...
            u = zeros(Float64, len, chunk_size + overlap_size)
            u[:, 1:locality] = train_seq[:, rear_overlap]
            u[:, locality+1 : locality+chunk_size] = train_seq[:, chunk_begin:chunk_end]
            u[:, locality+chunk_size+1 : end] = train_seq[:, forward_overlap]
            u .*= sigma
            
            # Construct Test Data (test_u)
            t_len = size(test_seq, 1)
            test_u = zeros(Float64, t_len, chunk_size + overlap_size)
            test_u[:, 1:locality] = test_seq[:, rear_overlap]
            test_u[:, locality+1 : locality+chunk_size] = test_seq[:, chunk_begin:chunk_end]
            test_u[:, locality+chunk_size+1 : end] = test_seq[:, forward_overlap]
            test_u .*= sigma
            
            # Reservoir Params
            approx_res_size = 5000
            nodes_per_input = round(Int, approx_res_size / (chunk_size + overlap_size))
            
            resparams = Dict(
                :N => nodes_per_input * (chunk_size + overlap_size),
                :train_length => 79000,
                :discard_length => 1000,
                :predict_length => 2999,
                :radius => 0.6,
                :degree => 3,
                :beta => 0.0001
            )
            resparams[:sparsity] = resparams[:degree] / resparams[:N]
            
            # Identify Neighbors for Communication
            my_p = myid()
            # We need the PID of front and rear workers relative to THIS worker
            # Since workers_list and channels are global in scope or passed explicitly?
            # They are captured by the closure if defined in main.
            
            # Logic to find neighbor PIDs:
            w_idx = findfirst(==(my_p), workers_list)
            f_pid = workers_list[mod(w_idx, numlabs) + 1]
            r_pid = workers_list[mod(w_idx - 2, numlabs) + 1]
            
            # Run Train & Predict
            # Transpose data because res_train_predict expects (InputDim x Time)
            result = res_train_predict(u', test_u', resparams, jobid, locality, chunk_size, 
                                     full_pred_marker_array, channels, f_pid, r_pid)
                                     
            return result
        end
        push!(futures, f)
    end
    
    # --- 결과 수집 ---
    results = fetch.(futures)
    
    # 결과를 공간축으로 병합 (Concatenate along dim 1)
    full_prediction = vcat(results...)
    
    println("Completed. Prediction size: ", size(full_prediction))
    return full_prediction
end

# 실행
# result = main_parallel_benchmark(4)