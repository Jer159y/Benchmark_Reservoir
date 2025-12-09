using Distributed

# addprocs(4)

@everywhere function heavy_calc(x)
    sleep(0.01)
    return x^2
end

# 1. 전체 데이터 준비
total_inputs = 1:1000
batch_size = 100

# 2. 결과 파일 초기화 (헤더 작성)
output_file = "batch_results.csv"
open(output_file, "w") do io
    println(io, "input,value")
end

println("Starting batch processing...")

# 3. Chunk 단위로 반복
# Iterators.partition을 사용하여 입력 데이터를 청크로 나눕니다.
for batch in Iterators.partition(total_inputs, batch_size)
    println("Processing batch: ", collect(batch))
    
    # A. 이번 배치만 병렬 처리 (결과가 100개만 옴)
    local_results = pmap(heavy_calc, batch)
    
    # B. 결과를 파일에 추가
    open(output_file, "a") do io
        for (input, value) in zip(batch, local_results)
            println(io, "$input,$value")
        end
    end

    # C. 메모리 정리
    # 루프가 돌면서 local_results는 덮어씌워지지만 명시적으로 비우고 싶은 경우 아래처럼 처리.
    local_results = nothing
    # 메모리가 아주 빡빡한 경우 가비지 컬렉터에게 청소 요청
    # GC.gc()
end