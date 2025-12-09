using Distributed

# Topology check
println("현재 프로세스 ID: ", myid())

if nprocs() == 1
    addprocs(2)
end

println("총 프로세스 수: ", nprocs())
println("모든 프로세스 ID: ", procs())

ids = pmap(id -> myid(), 1:10)
println("작업을 수행한 프로세스들: ", ids)


# Code Loading
function my_complex_job(x)
    return x * 2
end

# 실패 Case
try
    pmap(my_complex_job, 1:2)
catch e
    println("에러 발생: ", e)
end

# 성공 Case
@everywhere function my_complex_job(x)
    println("Worker $(myid()) is processing input: $x")
    return x * 2
end

# remotecall_fetch를 사용하여 특정 워커에서 작업 실행 (디버깅용)
result = remotecall_fetch(my_complex_job, 2, 10)
println("Remotecall fetch result from worker 2: ", result)

results = pmap(my_complex_job, 1:10)
println("Pmap results: ", results)


# Load Balancing
# A. pmap: 일이 무겁고 불규칙할 때 (동적 할당, 에러 추적 쉬움)
@everywhere function heavy_work(x)
    sleep(rand())
    return x^2
end

results_pmap = pmap(heavy_work, 1:20)

# B. @distributed: 일이 가볍고 균일할 때 (정적 할당, 에러 추적 어려움)
# (+), Reduction 연산자 부분을 빼면 계산만 하고 결과는 버림, return 값이 Task 객체로 나옴.
total_sum = @distributed (+) for i in 1:20
    i^2
end

println("Total sum using @distributed: ", total_sum)


# Data Transfer
big_data = rand(1000, 1000)

# 느린 방식
@everywhere function bad_func(idx)
    # Master에 있는 big_data를 매번 네트워크로 가져옴 (매우 느림)
    return sum(big_data[idx, :])
end

# 빠른 방식
# 미리 모든 Worker에 데이터를 미리 전송해 놓음, const 쓰면 더 좋음
@everywhere const shared_big_data = $big_data

@everywhere function good_func(idx)
    # 자기 메모리에 있는 local_data를 씀
    return sum(shared_big_data[idx, :])
end
# pmap(good_func, 1:1000)
