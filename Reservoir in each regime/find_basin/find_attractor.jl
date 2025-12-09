using Clustering
using Statistics

function extract_features(ts, xs)
    λ = compute_lambda(xs)              # Chaos indicator
    period = detect_period(ts, xs)    # Limit cycle indicator
    freqs = top_frequencies(xs)        # periodicity/quasi-periodicity indicator
    varx = var(xs[end-500:end])        # Variance of the tail
    mean_tail = mean(xs[end-500:end])  # Mean of the tail

    return vcat([λ, period, varx, mean_tail], freqs)
end

function dbscan_cluster(F; eps=0.5, minpts=4)
    # DBSCAN expects data as columns → F' (d × N)
    R = dbscan(F', eps, minpts)
    labels = R.assignments
    return labels
end

function discover_attractors_dbscan(initial_conditions, get_timeseries;
        eps=0.4, minpts=5)

    features = []
    for u0 in initial_conditions
        ts, xs = get_timeseries(u0)   # 모델 없이 데이터만 받는 함수
        push!(features, extract_features(ts, xs))
    end

    F = reduce(hcat, features)'   # N × d matrix

    labels = dbscan_cluster(F; eps=eps, minpts=minpts)

    return labels, F
end

initial_conditions = [(x,y) for x in -2:0.5:2, y in -2:0.5:2]

labels, F = discover_attractors_dbscan(initial_conditions, get_timeseries;
                                       eps=0.6, minpts=4)

println("각 초기조건의 attractor cluster = ", labels)
using Plots

xs = unique(x -> x[1], initial_conditions)
ys = unique(x -> x[2], initial_conditions)
nx, ny = length(xs), length(ys)

basin = reshape(labels, nx, ny)

heatmap(xs, ys, basin,
        xlabel="x0", ylabel="y0",
        title="DBSCAN-based Basin of Attraction")
