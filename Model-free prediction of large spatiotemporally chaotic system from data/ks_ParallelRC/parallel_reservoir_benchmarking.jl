using Distributed
using MAT
using JLD2 # 결과 저장용

# 워커 프로세스 추가 (예: 4개)
# 실제 실행 시에는 터미널에서 `julia -p 4`로 실행하거나 아래 줄 주석 해제
if nprocs() == 1
    addprocs(4)
end

@everywhere using SparseArrays, LinearAlgebra, Statistics, Random

# 모든 워커에 필요한 함수들을 로드
@everywhere include("generate_reservoir.jl")
@everywhere include("reservoir_layer.jl")
@everywhere include("train_reservoir.jl")
@everywhere include("fit.jl")
@everywhere include("indexing_function_forward.jl")
@everywhere include("indexing_function_rear.jl")
@everywhere include("synchronize.jl")
@everywhere include("predict.jl")
@everywhere include("res_train_predict.jl") # 아래 5번에서 정의

function parallel_reservoir_benchmarking(pool_size)
    # 데이터 경로 설정 (MATLAB 코드의 경로 참고)
    base_path = "./" # 현재 폴더로 가정
    
    # 인덱스 파일 로드
    # index_file = matread(joinpath(base_path, "testing_ic_indexes.mat"))
    # full_pred_marker_array = index_file["testing_ic_indexes"]
    # 테스트용 임시 마커
    full_pred_marker_array = [80000] 
    
    # 데이터 로드
    # train_data = matread(joinpath(base_path, "train_input_sequence.mat"))["train_input_sequence"]
    # test_data = matread(joinpath(base_path, "test_input_sequence.mat"))["test_input_sequence"]
    
    # (데모용) 임시 데이터 생성
    println("Generating dummy data for demo...")
    total_dim = 256
    train_data = rand(80000, total_dim)
    test_data = rand(20000, total_dim)
    
    # 통신 채널 생성 (모든 워커 간의 연결)
    # channels[(src, dst)] = Channel
    channels = Dict{Tuple{Int,Int}, RemoteChannel}()
    workers_list = workers()[1:pool_size] # 사용할 워커 ID 리스트
    
    for i in 1:pool_size
        my_w = workers_list[i]
        # 원형 연결 (Circular Topology)
        front_w = workers_list[mod(i, pool_size) + 1]       # i+1
        rear_w  = workers_list[mod(i - 2, pool_size) + 1]   # i-1
        
        # 채널 생성 (버퍼 크기 1 이상이어야 데드락 방지)
        channels[(my_w, front_w)] = RemoteChannel(()->Channel{Vector{Float64}}(10))
        channels[(my_w, rear_w)]  = RemoteChannel(()->Channel{Vector{Float64}}(10))
    end
    
    println("Starting Parallel Job on workers: $workers_list")
    
    # 각 워커에 작업 할당 (@spawnat)
    futures = []
    
    for (idx, pid) in enumerate(workers_list)
        # 각 워커에게 필요한 파라미터 전달
        f = @spawnat pid begin
            labindex = idx
            numlabs = pool_size
            jobid = 1
            
            # 파라미터 설정
            sigma = 0.5
            len, num_inputs_total = size(train_data)
            chunk_size = div(num_inputs_total, numlabs)
            
            chunk_begin = chunk_size * (labindex - 1) + 1
            chunk_end = chunk_size * labindex
            locality = 6
            
            # 오버랩 인덱스 계산
            rear_overlap = indexing_function_rear(chunk_begin, locality, num_inputs_total)
            forward_overlap = indexing_function_forward(chunk_end, locality, num_inputs_total)
            overlap_size = length(rear_overlap) + length(forward_overlap)
            
            # 데이터 분배 및 구성
            # MATLAB: u(:,1:locality) = ...
            u = zeros(Float64, len, chunk_size + overlap_size)
            u[:, 1:locality] = train_data[:, rear_overlap]
            u[:, locality+1 : locality+chunk_size] = train_data[:, chunk_begin:chunk_end]
            u[:, locality+chunk_size+1 : end] = train_data[:, forward_overlap]
            u = sigma .* u
            
            test_len = size(test_data, 1)
            test_u = zeros(Float64, test_len, chunk_size + overlap_size)
            test_u[:, 1:locality] = test_data[:, rear_overlap]
            test_u[:, locality+1 : locality+chunk_size] = test_data[:, chunk_begin:chunk_end]
            test_u[:, locality+chunk_size+1 : end] = test_data[:, forward_overlap]
            test_u = sigma .* test_u
            
            # Reservoir 파라미터
            approx_reservoir_size = 5000
            avg_degree = 3
            nodes_per_input = round(Int, approx_reservoir_size / (chunk_size + overlap_size))
            
            resparams = Dict(
                :N => nodes_per_input * (chunk_size + overlap_size),
                :train_length => 79000,
                :discard_length => 1000,
                :predict_length => 2999,
                :radius => 0.6,
                :degree => avg_degree,
                :beta => 0.0001
            )
            resparams[:sparsity] = avg_degree / resparams[:N]
            
            sync_length = 32
            
            # 예측 실행 (res_train_predict 호출)
            # channels를 인자로 전달해야 함
            pred_marker_array = [full_pred_marker_array[1]] # 데모용 하나만
            
            # 통신을 위한 이웃 ID 계산 (PID 기준)
            # 주의: workers_list 상의 인덱스가 아니라 실제 PID를 사용해야 함
            my_pid = myid()
            # worker_list 내에서의 내 위치 찾기
            list_idx = findfirst(==(my_pid), workers_list)
            front_pid = workers_list[mod(list_idx, numlabs) + 1]
            rear_pid = workers_list[mod(list_idx - 2, numlabs) + 1]
            
            pred_collect = res_train_predict(
                u', test_u', resparams, jobid, locality, chunk_size, 
                pred_marker_array, sync_length,
                channels, front_pid, rear_pid # 추가된 인자
            )
            
            return pred_collect
        end
        push!(futures, f)
    end
    
    # 결과 수집
    results = fetch.(futures)
    
    # 결과 병합 (gcat equivalent)
    # results는 Array of Arrays. 이를 하나의 큰 행렬로 합쳐야 함.
    # 각 result는 (chunk_size x time) 형태임.
    # 공간축(dim 1)으로 합침.
    full_prediction = vcat(results...) 
    
    println("Prediction complete. Shape: ", size(full_prediction))
    return full_prediction
end

# 실행 예시
# parallel_reservoir_benchmarking(4)