using DifferentialEquations
using GLMakie

"""
ρ<1	정지 (No Convection), 안정된 원점
1<ρ<24.74	안정된 흐름 (Steady Convection), 두 개의 안정된 고정점
ρ≈24.74	혼돈으로의 전환, 어트랙터 탄생
ρ>24.74	안정적인 혼돈 (Chaos), 이상한 어트랙터
"""

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

σ = 10.0
β = 8.0 / 3.0

u0 = [1.0, 1.0, 1.0]
t_simulation = (0.0, 500.0) # 충분한 시뮬레이션 시간
t_transient = 200.0       # 과도 상태를 버릴 시간

rho_range = 0:0.1:50

rho_vals = Float64[]
z_max_vals = Float64[]

for (i, ρ) in enumerate(rho_range)
    print("\rProgress: $(round(i/length(rho_range)*100, digits=1))% (ρ = $(round(ρ, digits=1)))")

    p = [σ, ρ, β]
    
    prob = ODEProblem(lorenz!, u0, t_simulation, p)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=0.02)

    start_index = findfirst(t -> t >= t_transient, sol.t)
    
    if isnothing(start_index) || start_index >= length(sol.t) - 1
        continue
    end

    for j in (start_index + 1):(length(sol.u) - 1)
        z_prev = sol.u[j-1][3]
        z_curr = sol.u[j][3]
        z_next = sol.u[j+1][3]
        
        if z_prev < z_curr > z_next
            push!(rho_vals, ρ)
            push!(z_max_vals, z_curr)
        end
    end
end

println("\nPlotting results...")

fig = Figure()
ax = Axis(fig[1, 1],
        xlabel = "ρ",
        ylabel = "z maxima",
        title = "Lorenz System Bifurcation Diagram")

scatter!(ax, rho_vals, z_max_vals,
                        markersize = 0.5,
                        color = :black)
vlines!(ax, [23.7], color=:red, label="ρ=23.7")

display(fig)

GLMakie.closeall()