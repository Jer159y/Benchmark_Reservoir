using DifferentialEquations, Makie, LinearAlgebra

# 로렌즈 시스템 함수 (이전과 동일)
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# 1. 파라미터 설정
σ = 10.0
β = 8.0 / 3.0
ρ = 23.5 # 23.7, 24.74, 25.0, 28.0


# 두 고정점 C+, C- 위치 계산 (아는 경우)
c_plus = [sqrt(β * (ρ - 1)), sqrt(β * (ρ - 1)), ρ - 1]
c_minus = [-sqrt(β * (ρ - 1)), -sqrt(β * (ρ - 1)), ρ - 1]

# 2. 격자 설정
resolution = 300
x_range = range(-30, 30, length=resolution)
y_range = range(-30, 30, length=resolution)
z0 = ρ - 1  # z값 고정

# 결과를 저장할 행렬 (1: C+ 베이슨, -1: C- 베이슨, 0: 기타)
basin_matrix = zeros(Int, resolution, resolution)

# 시뮬레이션 설정
t_sim = (0.0, 200.0)
tolerance = 1.0 # 어트랙터에 수렴했는지 판단할 거리 임계값

# 3 & 4. 모든 격자점에서 시뮬레이션 및 판별
for (i, x0) in enumerate(x_range)
    println("Processing row $i of $resolution...")
    for (j, y0) in enumerate(y_range)
        u0 = [x0, y0, z0]
        prob = ODEProblem(lorenz!, u0, t_sim, [σ, ρ, β])
        sol = solve(prob, Tsit5(), save_start=false, save_end=true)
        
        u_final = sol.u[end]

        # 최종 위치가 어느 어트랙터에 가까운지 판별
        if norm(u_final - c_plus) < tolerance
            basin_matrix[i, j] = 1
        elseif norm(u_final - c_minus) < tolerance
            basin_matrix[i, j] = -1
        else
            basin_matrix[i, j] = 0
        end
    end
end

# 5. 시각화
println("Plotting...")
heatmap(x_range, y_range, basin_matrix', # 행렬을 전치해야 축이 맞음
        title="Basin of Attraction for Lorenz System (ρ=$ρ, z₀=$z0)",
        xlabel="x-axis", ylabel="y-axis",
        c=cgrad([:red, :black, :blue]), # -1: red, 0: black, 1: blue
        aspect_ratio=:equal
)