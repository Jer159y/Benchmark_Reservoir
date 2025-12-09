"""
1st file

picked ρ values: 23.0(fixed point), 28.0(chaotic), 100.0(periodic)

reference bifurcation points:
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

function lorenz_jac!(J, u, p, t)
    σ, ρ, β = p
    J[1, 1] = -σ
    J[1, 2] = σ
    J[1, 3] = 0.0
    J[2, 1] = ρ - u[3]
    J[2, 2] = -1.0
    J[2, 3] = -u[1]
    J[3, 1] = u[2]
    J[3, 2] = u[1]
    J[3, 3] = -β
    return nothing
end

function gen_lorenz_data(ρ_values, u0s, t_simulation, t_transient)
    data_used = []
    
    for (ρ, u0) in Iterators.product(ρ_values, u0s)
        print("\rProgress: ρ=$(round(ρ, digits=1)), u0=$(u0)")
        p = [10.0, ρ, 8.0 / 3.0]
        prob = ODEProblem(lorenz!, u0, t_simulation, p)
        sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=0.02)
        push!(data_used, (ρ=ρ, u0=u0, t=sol.t, u=hcat(sol.u...)))
    end
    return data_used
end

function cal_lyapunov_analytic(ρ_values, u0, t_simulation, t_transient)
    λs = []

    for ρ in ρ_values
        p = [10.0, ρ, 8.0 / 3.0]
        ds = ContinuousDynamicalSystem(lorenz!, u0, p)
        ds = TangentDynamicalSystem(ds; J=lorenz_jac!)
        λ = lyapunovspectrum(ds, round(Int, (t_simulation[2]-t_simulation[1])/0.02); Δt=0.02, Ttr=t_transient*0.02)
        push!(λs, (ρ=ρ, λ=λ))
    end
    return λs
end

ρs = [28.0] # [12.0, 23.0, 28.0, 100.0]

t_simulation = (0.0, 500.0)
t_transient = 200.0

u0s = [
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.05],
    [0.5, 0.5, 0.5],
    [10.0, 10.0, 10.0],
    [-5.0, -5.0, 25.0],
    [0.1, 0.1, 0.1],
    [2.0, 3.0, 4.0],
    [-1.0, -1.0, -1.0],
    [5.0, 5.0, 5.0],
    [-10.0, 10.0, 10.0]
]

data_used = gen_lorenz_data(ρs, u0s, t_simulation, t_transient)
# λs = cal_lyapunov_analytic(ρs, u0s[1], t_simulation, t_transient)


"""
u0 = [1.0, 1.0, 1.0]
rho_used = [23.0, 28.0, 100.0]
rho_range = 0:0.1:110

rho_vals_z = Float64[]
z_max_vals = Float64[]
rho_vals_x = Float64[]
x_max_vals = Float64[]

for (i, ρ) in enumerate(rho_used) # or rho_used
    print("\rProgress: $(round(i/length(rho_used)*100, digits=1))% (ρ = $(round(ρ, digits=1)))")

    p = [σ, ρ, β]
    
    prob = ODEProblem(lorenz!, u0, t_simulation, p)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=0.02)

    start_index = findfirst(t -> t >= t_transient, sol.t)
    
    if isnothing(start_index) || start_index >= length(sol.t) - 1
        continue
    end

    if ρ in rho_used
        ds = ContinuousDynamicalSystem(lorenz!, u0, p)
        ds = TangentDynamicalSystem(ds; J=lorenz_jac!)
        λ = lyapunovspectrum(ds, round(Int, (t_simulation[2]-t_simulation[1])/0.02); Δt=0.02, Ttr=t_transient*0.02)
        push!(data_used, (ρ=ρ, t=sol.t, u=hcat(sol.u...)))
        push!(λs, (ρ=ρ, λ=λ))
    end

    for j in (start_index + 1):(length(sol.u) - 1)
        z_prev = sol.u[j-1][3]
        z_curr = sol.u[j][3]
        z_next = sol.u[j+1][3]

        x_prev = sol.u[j-1][1]
        x_curr = sol.u[j][1]
        x_next = sol.u[j+1][1]
        
        if z_prev < z_curr > z_next
            push!(rho_vals_z, ρ)
            push!(z_max_vals, z_curr)
        end
        if x_prev < x_curr > x_next
            push!(rho_vals_x, ρ)
            push!(x_max_vals, x_curr)
        end
    end
end
"""
# fig1 = plot_bifurcation_diagram(rho_vals_x, x_max_vals, rho_vals_z, z_max_vals, data_used)
# fig2 = plot_phase_space(data_used)
