# Reservoir state space에서의 Basin of Attraction 분석을 위한 코드
# 계산량이 너무 커서 안돌아감...
# u = x(t), p = W

function reservoir_dynamics(out, u, p, t)
    W = p
    out = tanh.(W * u)
    return nothing
end

p_reservoir = W_new
u0 = zeros(1500)
u0[1] = 0.1; u0[2] = 0.1;
ds_reservoir = DiscreteDynamicalSystem(reservoir_dynamics, u0, p_reservoir)

T = 10000 # N의 몇 배수로 설정
λs = lyapunov_spectrum(ds_reservoir, T; dt=1, method=:QR, maxiter=1000)
λ_max = maximum(λs)
println("Reservoir Lyapunov Exponent λ_max: ", λ_max)

plane_indices = (1, 2)    # 스캔할 차원 i, j
x1_range = -2.0:0.05:2.0  # x_1 축 범위
x2_range = -2.0:0.05:2.0  # x_2 축 범위
fixed = ntuple(_ -> range(0.0; length=1), 1500-2)
grid = (x1_range, x2_range)
grid = (x1_range, x2_range, fixed...)
println("Calculating basins... (N=$(length(u0)) 차원 시스템이라 오래 걸릴 수 있습니다)")
basins, attractors = basins_of_attraction(grid, ds_reservoir; 
                                        plane = plane_indices,
                                        Ttr = 100)

println("Calculation complete.")
println("Found $(length(attractors)) attractors on this plane.")

p = heatmap(x1_range, x2_range, basins'; # ' (transpose)를 사용하여 축을 맞춥니다.
        xlabel="State x[$(plane_indices[1])]",
        ylabel="State x[$(plane_indices[2])]",
        title="Reservoir Internal Dynamics Basin (N=$N, ρ(W)=$target_spectral_radius)",
        aspect_ratio=:equal,
        legend=:none,
        c=:darktest) # 색상 맵

for (key, att) in attractors
    # att는 N차원 끌개 데이터. 여기서 (1, 2) 차원 값만 추출
    att_point = [att[plane_indices[1]], att[plane_indices[2]]]
    
    # ChaosTools 5.0 이상에서는 att가 Dataset이므로 평균값을 사용
    if att isa Dataset
        att_point = [mean(att[:,plane_indices[1]]), mean(att[:,plane_indices[2]])]
    end

    scatter!(p, [att_point[1]], [att_point[2]], 
             markersize=5, markercolor=:white, markerstrokecolor=:black)
end