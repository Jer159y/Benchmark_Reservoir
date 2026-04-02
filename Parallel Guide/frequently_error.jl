# Random Number Generation
# 문제: Simulation을 병렬로 100번 돌릴 때, 이 결과가 모두 똑같거나 통계적으로 이상할 수 있음.
# 해결: 모든 Worker가 같은 Seed로 초기화되기 때문, 서로 다른 시드를 갖도록 초기화해야 함.
@everywhere using Random

@everywhere Random.seed!(1234 + myid())  # 워커 ID를 더해서 시드 설정


# 파일 입출력(File I/O) (Race Condition)
# 문제: 여러 Worker가 동시에 같은 파일에 쓰기 시도 -> 파일 손상 가능성, 데이터 섞임, 충돌 등.
# 해결: 각 워커가 고유한 파일 이름을 사용하거나, 파일 쓰기 작업을 Master에서만 수행하도록 설계해야 함. (후자 추천)
@everywhere function write_unique_file(data, worker_id)
    filename = "output_worker_$(worker_id).txt"
    open(filename, "w") do io
        write(io, "Data from worker $(worker_id):\n")
        write(io, string(data))
    end
end
# pmap(i -> write_unique_file(rand(5), myid()), 1:4)


# Type Stability & Serialization Error
# 문제: Master에서 정의한 복잡한 객체(Struct)를 Worker에게 보내는 중에 에러 발생.
# 원인 1(Type Stability): 객체 필드가 불안정한 타입을 가짐. @everywhere로 구조체(Struct) 정의를 안해줬기 때문.
# 원인 2(Serialization): 객체가 직렬화 불가능한 필드를 포함. 어떤 데이터(예: IOStream, GPU 배열 등)는 직렬화 불가능.
# 해결: Worker에게는 순수한 데이터(숫자, 문자열, 배열, 잘 정의된 Struct 등)만 보내도록 설계. 파일 포인터나 DB 연결 같은 것은 넘기지 말고, 필요한 경우 Worker에서 새로 생성.

# 원인 1
# Error 방식
struct MyParam
    a::Int
    b::Float64
end

function worker_task(p::MyParam)
    return p.a + p.b
end

data = MyParam(10, 20.5)

try 
    results = pmap(worker_task, [data, data])
catch e
    println("에러 발생: ", e)
end

# Fix 방식
@everywhere struct MyParam
    a::Int
    b::Float64
end

@everywhere function worker_task(p::MyParam)
    return p.a + p.b
end

data = MyParam(10, 20.5)
println(remotecall_fetch(worker_task, 2, data))  # Worker 2에서 test

# 원인 2
# Error 방식
# Master에서 파일 열기
file_handel = open("somefile.txt", "w")

@everywhere function write_to_file(f_hanlde, data)
    # Worker가 Master의 파일 핸들을 사용하려고 시도
    write(f_hanlde, data)
end

try
    remotecall_fetch(write_to_file, 2, file_handel, "Hello from worker 2\n")
catch e
    println("에러 발생: ", e)
end
close(file_handel)

# Fix 방식
# 파일 객체(IOStream 등)는 직렬화 불가능하므로 Worker에 넘기지 말고, Worker가 직접 파일을 열도록 함.
@everywhere function safe_write_to_file(filename, data)
    open(filename, "a") do io
        write(io, "Worker $(myid()) says: $data\n")
    end
    return true
end

pmap(i -> safe_write_to_file("safe_output.txt", "Hello from worker $i"), 1:4)

# 데이터 전송이 가능한지 Master에서 미리 테스트
using Serialization

suspect_data = open("safe_output.txt", "w") # IOStream 객체
# suspect_data = rand(100)               # 직렬화 가능한 객체

buf = IOBuffer()

try
    serialize(buf, suspect_data)
    println("데이터는 직렬화 가능합니다.")
catch e
    println("데이터 직렬화 실패: ", e)
end

# 데이터 전송이 가능한지 작게 미리 테스트

f_handle = open("somefile.txt", "w")
try
    # Identity function test
    returned_handel = remotecall_fetch(x -> x, workers()[1], f_handle)
    println("파일 핸들 전송 성공: ", returned_handel)

    # But IOStream은 여기서 성공이 떠도, 써보면 에러남. Worker 입장에서 열려있는지 확인.
    is_open = remotecall_fetch(x -> isopen(x), workers()[1], f_handle)
    println("Worker가 본 파일 상태: ", is_open ? "열려있음" : "닫혀있음")
catch e
    println("파일 핸들 전송 실패: ", e)
end

# 해당 코드는 통과했지만, 사실 IOStream, Ptr (C 포인터), GPU 배열, Task/Channel, Database Connection 등은 직렬화 불가능하므로 주의해야 함.


# Master에서는 구조체를 수정(재정의)했으나 Worker는 옛날 정의를 가지고 있는 경우 난해한 직렬화 에러 발생.
# 즉, 구조체 정의가 바뀌면 세션을 껐다가 켜는 것이 가장 좋음.


