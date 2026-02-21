using FFTW
using LinearAlgebra
using Statistics

function kursiv_solve(init, ModelParams)
    N = Int(ModelParams[:N])
    d = Float64(ModelParams[:d])
    
    # Julia는 range에서 collect를 사용하여 배열로 변환합니다.
    x = d .* collect(-N/2+1:N/2) ./ N
    u = init  # 입력이 이미 1차원 배열이라고 가정
    v = fft(u)
    
    h = ModelParams[:tau]
    
    # k 벡터 생성 (MATLAB의 [0:N/2-1 0 -N/2+1:-1]' 대응)
    k = vcat(0:N/2-1, 0, -N/2+1:-1) .* (2*pi/d)
    
    # L = k.^2 - k.^4 (element-wise)
    L = k.^2 .- k.^4
    
    E = exp.(h .* L)
    E2 = exp.(h .* L ./ 2)
    
    M = 16
    r = exp.(1im * pi * ((1:M) .- 0.5) ./ M)
    
    # LR 계산: 브로드캐스팅을 사용하여 MATLAB의 ones 확장과 유사하게 처리
    # L은 (N,), r은 (M,) -> LR은 (N, M)
    LR = h .* L .+ r' 
    
    # 계수 계산 (mean over dim 2)
    # LR이 작을 때 수치 안정성을 위해 정규화
    eps_safe = 1e-10
    LR_safe = LR .+ eps_safe .* (abs.(LR) .< eps_safe)
    
    Q_raw = real.(mean((exp.(LR_safe./2) .- 1) ./ LR_safe, dims=2))
    f1_raw = real.(mean((-4 .- LR_safe .+ exp.(LR_safe).*(4 .- 3 .* LR_safe .+ LR_safe.^2)) ./ (LR_safe.^3 .+ eps_safe), dims=2))
    f2_raw = real.(mean((2 .+ LR_safe .+ exp.(LR_safe).*(-2 .+ LR_safe)) ./ (LR_safe.^3 .+ eps_safe), dims=2))
    f3_raw = real.(mean((-4 .- 3 .* LR_safe .- LR_safe.^2 .+ exp.(LR_safe).*(4 .- LR_safe)) ./ (LR_safe.^3 .+ eps_safe), dims=2))
    
    Q  = h .* Q_raw
    f1 = h .* f1_raw
    f2 = h .* f2_raw
    f3 = h .* f3_raw
    
    # 차원 축소 (Nx1 행렬을 N 벡터로 변환)
    Q = dropdims(Q, dims=2)
    f1 = dropdims(f1, dims=2)
    f2 = dropdims(f2, dims=2)
    f3 = dropdims(f3, dims=2)
    
    # 계수 검증
    if !all(isfinite, [Q; f1; f2; f3])
        error("Non-finite coefficients in ETD: Q=$(count(!isfinite, Q)), f1=$(count(!isfinite, f1)), f2=$(count(!isfinite, f2)), f3=$(count(!isfinite, f3))")
    end
    
    nmax = ModelParams[:nstep]
    println("KS solve: tau=$(h), N=$(N), d=$(d), nstep=$(nmax)")
    
    g = -0.5im .* k
    
    vv = zeros(ComplexF64, N, nmax)
    vv[:, 1] = v
    
    for n = 1:nmax
        # t = n * h (사용되지 않음)
        
        # 비선형 항 계산 로직
        Nv = g .* fft(real.(ifft(v)).^2)
        a = E2 .* v .+ Q .* Nv
        
        Na = g .* fft(real.(ifft(a)).^2)
        b = E2 .* v .+ Q .* Na
        
        Nb = g .* fft(real.(ifft(b)).^2)
        c = E2 .* a .+ Q .* (2 .* Nb .- Nv)
        
        Nc = g .* fft(real.(ifft(c)).^2)
        v = E .* v .+ Nv .* f1 .+ 2 .* (Na .+ Nb) .* f2 .+ Nc .* f3
        
        # 수치 체크
        if any(!isfinite, v)
            error("Non-finite values in v at time step n=$n (of $nmax)")
        end
        
        # 크기 체크 (발산 방지) - 1%마다만 체크해서 로그 감소
        max_v = maximum(abs.(v))
        if max_v > 1e8
            if mod(n, max(1, div(nmax, 100))) == 0  # 1%마다 한번
                @warn "Large magnitude detected at step $n: max|v|=$max_v (실패 위험)"
            elseif max_v > 1e10
                error("Solver diverged: max|v|=$max_v at step $n. 시간 단계를 줄여야 합니다.")
            end
        end
        
        vv[:, n] = v
    end
    
    uu = real.(ifft(vv, 1)) # 열 방향 IFFT
    return transpose(uu) # MATLAB 코드와 동일한 형태(Time x Space)로 반환
end