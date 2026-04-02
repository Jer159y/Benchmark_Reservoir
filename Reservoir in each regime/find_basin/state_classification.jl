
using FFTW


using DynamicalSystems

"""
compute_lambda(xs, τ, m)

입력:
  xs: time-series data
  τ: embedding delay (Embedding 자동 최적화 필요, AMI로 delay 선택, FNN으로 dimension 선택 권장)
  m: embedding dimension
출력:
  λ: largest Lyapunov exponent estimate
"""
function compute_lambda(xs; τ=10, m=3)
    embedded = embed(Dataset(xs), m, τ)
    return lyapunov(embedded)
end

# dominant frequencies
function top_frequencies(xs; k=3)
    f = abs.(fft(xs))
    idx = sortperm(f, rev=true)[1:k]
    return float.(idx)
end

# period via peak detection
function detect_period(ts, xs; min_prom=0.05, win=5)
    peaks_t = Float64[]
    for i in 2:length(xs)-1
        if xs[i] > xs[i-1] && xs[i] > xs[i+1] &&
           xs[i] - min(xs[i-1], xs[i+1]) > min_prom
            push!(peaks_t, ts[i])
        end
    end

    if length(peaks_t) < win
        return 0.0
    end

    periods = diff(peaks_t)
    return mean(periods[end-win+1:end])
end

function detect_quasiperiodic(xs; tol_ratio=0.1)
    f = abs.(fft(xs))
    peaks = findall(i -> f[i] > maximum(f)*tol_ratio, 1:length(f))

    if length(peaks) == 1
        return :periodic
    else
        return :quasiperiodic
    end
end

function classify_attractor_from_data(ts, xs)

    # 1) limit cycle test
    lc_type, period = detect_limit_cycle(ts, xs)
    if lc_type == :limit_cycle
        return (:limit_cycle, period)
    end

    # 2) Lyapunov exponent
    λ = compute_lambda(xs)
    if λ > 0.05
        return (:chaotic, λ)
    elseif abs(λ) < 0.01
        # need to distinguish periodic vs quasi-periodic
        pq = detect_quasiperiodic(xs)
        return (pq, λ)
    else
        return (:fixed_point, λ)
    end
end
