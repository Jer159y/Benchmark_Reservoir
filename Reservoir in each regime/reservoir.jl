using ReservoirComputing
using LinearAlgebra

function ESN_task(input_data, target_data, predict_len)
    res_size = 1500
    in_size = 3
    res_radius = 1.2
    res_sparsity = 30 / 1500
    input_scaling = 0.1

    esn = ESN(input_data, in_size, res_size;
        reservoir=rand_sparse(; radius=res_radius, sparsity=res_sparsity),
        input_layer=weighted_init(; scaling=input_scaling),
        reservoir_driver=RNN(),
        nla_type=NLADefault(),
        states_type=StandardStates())
    training_method = StandardRidge(1e-6)

    output_layer = train(esn, target_data, training_method)
    output = esn(Generative(predict_len), output_layer)
    return esn, output_layer, output
end

function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

σ = 10.0
β = 8.0 / 3.0
ρs = [23.5, 23.7, 24.0, 24.74, 25.0, 28.0]

u0 = [1.0, 1.0, 1.0]
t_simulation = (0.0, 500.0)
t_transient = 200.0

shift = 300
train_len = 5000
predict_len = 1250

test_datas = []
outputs = []
esns = []
readouts = []

for ρ in ρs
    p = [σ, ρ, β]

    prob = ODEProblem(lorenz!, u0, t_simulation, p)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-6, saveat=0.02)

    data = hcat(sol.u...)

    input_data = data[:, shift:(shift + train_len - 1)]
    target_data = data[:, (shift + 1):(shift + train_len)]
    test_data = data[:, (shift + train_len + 1):(shift + train_len + predict_len)]
    push!(test_datas, test_data)

    esn, readout, output = ESN_task(input_data, target_data, predict_len)
    push!(outputs, output)
    push!(esns, esn)
    push!(readouts, readout)
end

println(propertynames(esns[1]), ", ", propertynames(readouts[1]))

esn = esns[1]
W_in = esn.input_matrix
W_res = esn.reservoir_matrix

readout = readouts[1]
W_out = readout.output_matrix

W = W_res + W_in * W_out
ρ_W = maximum(abs.(eigvals(W)))



fig = Figure(size=(2000, 900), fontsize=14)
axs = [Axis(fig[i, j]) for i in 1:3, j in 1:length(ρs)]

ts = 0.0:0.02:25.0
lorenz_maxlyap = 0.9056
lyap_time = (0:predict_len-1) .* 0.02 ./ lorenz_maxlyap

coords = ["x(t)", "y(t)", "z(t)"]

for (j, ρ) in enumerate(ρs)
    test_data = test_datas[j]
    output = outputs[j]

    for i in 1:3
        ax = axs[i, j]
        lines!(ax, lyap_time, test_data[i, :], color=:black, linewidth=2.0, label="actual")
        lines!(ax, lyap_time, output[i, :], color=:red, alpha=0.5, linewidth=2.0, label="predicted")

        if j == 1
            ax.ylabel = coords[i]
        end
        if i == 3
            ax.xlabel = "max(λ)*t"
        end
        if i == 1
            ax.title = "ρ = $(ρ)"
        end

        if i == 1
            ax.yticks = -15:15:15
        elseif i == 2
            ax.yticks = -20:20:20
        else
            ax.yticks = 10:15:40
        end

        axislegend(ax, position=:rt)
    end
end

Label(fig[0, 1:length(ρs)], "Lorenz System Coordinates", tellwidth=false)

display(fig)
GLMakie.closeall()

