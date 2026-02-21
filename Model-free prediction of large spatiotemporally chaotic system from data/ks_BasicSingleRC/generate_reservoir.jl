using SparseArrays
using LinearAlgebra
using Arpack

function generate_reservoir(size::Int, radius::Float64, degree::Int)
    sparsity = degree / size
    
    # 희소 행렬 생성 (sprand)
    A = sprand(size, size, sparsity)
    
    # 스펙트럼 반경 계산 (eigs)
    # Arpack의 eigs는 (values, vectors) 튜플 등을 반환합니다.
    # ritzvec=false는 고유벡터 계산을 생략하여 속도를 높입니다.
    vals, _ = eigs(A; nev=1, which=:LM, ritzvec=false)
    e = maximum(abs.(vals))
    if !isfinite(e) || e == 0
        error("Invalid spectral radius computed (" * string(e) * ")")
    end
    
    # 스펙트럼 반경 조정
    A = (A ./ e) .* radius
    
    return A
end