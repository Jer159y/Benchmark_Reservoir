using Distributed

# 1. 환경 설정 (Master)
if nprocs() == 1
    addprocs(3)
end

println("Number of processes: ", nworkers()) # nprocs = 4 (1 Master + 3 Workers)

# 2. Worker 훈련 (@everywhere)
@everywhere begin
    using Random
    
    struct ExperimentalResults
        id::Int
        value::Float64
        status::String
    end

    function run_simulation(id::Int)
        sleep(0.1+rand()*0.4)
        val = rand() * 100
        if val > 50
            return ExperimentalResults(id, val, "Success")
        else
            return ExperimentalResults(id, val, "Failure")
        end
    end
end

# 3. Master에서 작업 분배 및 결과 수집 (Map-Reduce, Scatter-Gather)
println("Distributing tasks to workers...")

input_ids = collect(1:20)  # 20개의 시뮬레이션 작업

# pmap은 결과를 순서대로 배열(Vector)에 담아 Master로 반환
# Worker -> Master로 데이터 이동 (직렬화 가능한 Struct만 가능)
results = pmap(run_simulation, input_ids)

println("All simulations completed. Results:")
for res in results
    println("ID: ", res.id, ", Value: ", res.value, ", Status: ", res.status)
end

# 4. 최종 작업 (Master)
# 파일 쓰기, DB 저장, 통계 분석 등
output_file = "file_report.txt"

open(output_file, "w") do io
    total_success = 0
    println(io, "Simulation Results Summary")
    for res in results
        println(io, "ID: ", res.id, ", Value: ", res.value, ", Status: ", res.status)
        if res.status == "Success"
            total_success += 1
        end
    end
    println(io, "Total Successful Simulations: ", total_success)
end