using FFTW
using LinearAlgebra
using Statistics
using MAT # .mat 파일 저장을 위해 필요 (add MAT)

function generate_data_asym()
    N = 128
    d = 60
    x = d .* collect(-N/2+1:N/2) ./ N
    
    delta = 0.0
    wavelength = d/4
    omega = 2*pi/wavelength
    
    p = delta .* cos.(omega .* x)
    px = -omega * delta .* sin.(omega .* x)
    pxx = -(omega^2) .* p
    
    u = 0.6 .* (-1 .+ 2 .* rand(size(x, 1)))
    v = fft(u)
    
    h = 1/4
    k = vcat(0:N/2-1, 0, -N/2+1:-1) .* (2*pi/d)
    L = k.^2 .- k.^4
    
    E = exp.(h .* L)
    E2 = exp.(h .* L ./ 2)
    
    M = 16
    r = exp.(1im * pi * ((1:M) .- 0.5) ./ M)
    LR = h .* L .+ r'
    
    Q  = h .* real.(mean((exp.(LR./2) .- 1) ./ LR, dims=2))
    f1 = h .* real.(mean((-4 .- LR .+ exp.(LR).*(4 .- 3 .* LR .+ LR.^2)) ./ LR.^3, dims=2))
    f2 = h .* real.(mean((2 .+ LR .+ exp.(LR).*(-2 .+ LR)) ./ LR.^3, dims=2))
    f3 = h .* real.(mean((-4 .- 3 .* LR .- LR.^2 .+ exp.(LR).*(4 .- LR)) ./ LR.^3, dims=2))
    
    Q = dropdims(Q, dims=2)
    f1 = dropdims(f1, dims=2)
    f2 = dropdims(f2, dims=2)
    f3 = dropdims(f3, dims=2)
    
    tt = 0
    tmax = 25000
    nmax = round(Int, tmax/h)
    
    g = -0.5im .* k
    
    vv = zeros(ComplexF64, N, nmax)
    vv[:, 1] = v
    
    for n = 1:nmax
        # t = n*h
        rifftv = real.(ifft(v))
        
        # 비대칭 항이 추가된 비선형 부분
        Nv = g .* fft(rifftv.^2) .+ 2im .* k .* fft(rifftv .* px) .- fft(rifftv .* pxx) .+ k.^2 .* fft(rifftv .* p)
        
        a = E2 .* v .+ Q .* Nv
        riffta = real.(ifft(a))
        Na = g .* fft(riffta.^2) .+ 2im .* k .* fft(riffta .* px) .- fft(riffta .* pxx) .+ k.^2 .* fft(riffta .* p)
        
        b = E2 .* v .+ Q .* Na
        rifftb = real.(ifft(b))
        Nb = g .* fft(rifftb.^2) .+ 2im .* k .* fft(rifftb .* px) .- fft(rifftb .* pxx) .+ k.^2 .* fft(rifftb .* p)
        
        c = E2 .* a .+ Q .* (2 .* Nb .- Nv)
        rifftc = real.(ifft(c))
        Nc = g .* fft(rifftc.^2) .+ 2im .* k .* fft(rifftc .* px) .- fft(rifftc .* pxx) .+ k.^2 .* fft(rifftc .* p)
        
        v = E .* v .+ Nv .* f1 .+ 2 .* (Na .+ Nb) .* f2 .+ Nc .* f3
        vv[:, n] = v
    end
    
    uu = real.(ifft(vv, 1))' # Transpose to match MATLAB output
    
    # 결과 저장 (.mat)
    filename = "mult_asym_kursiv_delta$(Int(100*delta))wl$(wavelength).mat"
    matwrite(filename, Dict("uu" => collect(uu), "d" => d, "wavelength" => wavelength))
    println("Saved $filename")
    
    # 시각화
    heatmap(uu, c=:jet)
end