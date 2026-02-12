"""
2nd file
"""

function train_test_split(data, shift, washout, train_len, predict_len)
    input_data = data[:, shift:(shift + train_len - 1)]
    target_data = data[:, (washout + shift + 1):(shift + train_len)]
    test_data = data[:, (shift + train_len + 1):(shift + train_len + predict_len)]
    return input_data, target_data, test_data
end

function gen_trained_data(data; shift=1, washout=1000, train_len=5000, predict_len=1250, 
                          res_size=1000, rng=MersenneTwister(42))
    args = HyperParams(res_size, 0.9, 30 / 1500, 0.1, 0.4, washout)
    training_method = StandardRidge(1e-6)

    input_data, target_data, test_data = train_test_split(data, shift, washout, train_len, predict_len)
    esn_param = standardParam(input_data, args)
    esn_param[:initial_state] = zeros(args.res_size)
    esn = generate_esn(esn_param, rng)
    output_layer = train(esn, target_data, training_method)
    
    return (input_data=input_data, target_data=target_data, test_data=test_data,
            esn=esn, readout=output_layer, param=esn_param)
end

function run_closed_prediction(data; shift=1, washout=1000, train_len=5000, predict_len=1250, 
                                res_size=1500, rng=MersenneTwister(42))
    
    trained = gen_trained_data(data; shift=shift, washout=washout, train_len=train_len, 
                                predict_len=predict_len, res_size=res_size, rng=rng)
    predictive_output = trained.esn(Predictive(trained.test_data), trained.readout)
    closed_output = trained.esn(Generative(predict_len), trained.readout)

    test_start = shift + train_len + 1
    test_range = test_start:(test_start + predict_len - 1)
    test_time = collect(test_range)

    return (trained...,
            test_time=test_time, predictive_output=predictive_output, closed_output=closed_output)
end

function build_closed_prediction_runs(datas; shift=1, washout=1000, train_len=5000, predict_len=1250, rng=MersenneTwister(42))
    runs = []
    for data in datas
        push!(runs, run_closed_prediction(data; shift=shift, washout=washout, train_len=train_len,
                                            predict_len=predict_len, rng=rng))
    end
    return runs
end

function plot_closed_predictions(base_run, other_runs)
    fig = Figure(size=(1200, 900))
    coords = ["x(t)", "y(t)", "z(t)"]
    palette = [:red, :blue, :green, :purple, :orange, :teal, :brown, :magenta]

    for i in 1:3
        left_ax = Axis(fig[i, 1], ylabel=coords[i])
        lines!(left_ax, base_run.test_time, base_run.test_data[i, :], color=:black, label="target/test")
        lines!(left_ax, base_run.test_time, base_run.closed_output[i, :], color=:red, alpha=0.7, label="closed pred")
        scatter!(left_ax, base_run.test_time[1], base_run.test_data[i, 1], color=:black, markersize=8)
        if i == 1
            left_ax.title = "Base start (ρ=$(base_run.ρ))"
            axislegend(left_ax, position=:rt)
        elseif i == 3
            left_ax.xlabel = "time"
        end

        right_ax = Axis(fig[i, 2], ylabel=coords[i])
        if i == 1
            right_ax.title = "Closed preds from other starts"
        elseif i == 3
            right_ax.xlabel = "time"
        end

        for (j, run) in enumerate(other_runs)
            color = palette[(j - 1) % length(palette) + 1]
            label = "ρ=$(round(run.ρ, digits=2)), u0=$(round.(run.u0; digits=2))"
            lines!(right_ax, run.test_time, run.closed_output[i, :], color=color, label=label)
            scatter!(right_ax, run.test_time[1], run.closed_output[i, 1], color=color, markersize=6)
        end

        if i == 1 && !isempty(other_runs)
            axislegend(right_ax, position=:rt, nbanks=2)
        end
    end

    return fig
end



# rng = rand(Int); println("Random seed: ", rng)
rng = MersenneTwister(42)

p = (N, I_wave, K_wave)
prob = ODEProblem(hr_net!, u0, tspan, p)
sol = solve(prob, Tsit5(), saveat=0.5, reltol=1e-4)
data = reduce(hcat, [[sol.u[t][3i-2] for i in 1:N] for t in 1:length(sol.t)])
results = run_closed_prediction(data; train_len=5000, predict_len=5000, res_size=10000, rng=rng)

fig = Figure(size=(1200, 900))

ax1 = Axis(fig[1, 1], title="Training Input Data", xlabel="Time", ylabel="Neuron Index")
lines!(ax1, sol.t, 1:N, data[:, 1:size(results.input_data, 2)]', colormap=:magma)
ax2 = Axis(fig[1, 2], title="Training Target Data", xlabel="Time", ylabel="Neuron Index")
lines!(ax2, sol.t, 1:N, results.target_data', colormap=:magma)
ax3 = Axis(fig[2, 1], title="Open-loop Prediction", xlabel="Time", ylabel="Neuron Index")
lines!(ax3, results.test_time, 1:N, results.predictive_output', colormap=:magma)
ax4 = Axis(fig[2, 2], title="Closed-loop Prediction", xlabel="Time", ylabel="Neuron Index")
lines!(ax4, results.test_time, 1:N, results.closed_output', colormap=:magma)

display(fig)

fig = Figure(size=(1200, 400 * N + 10))

Label(fig[1, 1, Top()], "Training Input", fontsize=14, font=:bold, padding=(0, 0, 20, 0))
Label(fig[1, 2, Top()], "Training Target", fontsize=14, font=:bold, padding=(0, 0, 20, 0))
Label(fig[1, 3, Top()], "Open-loop Prediction", fontsize=14, font=:bold, padding=(0, 0, 20, 0))
Label(fig[1, 4, Top()], "Closed-loop Prediction", fontsize=14, font=:bold, padding=(0, 0, 20, 0))

input_time = sol.t[1:size(results.input_data, 2)]
target_time = sol.t[1:size(results.target_data, 2)]

for i in 1:N
    # Training Input
    ax1 = Axis(fig[i, 1], ylabel="Neuron $i")
    lines!(ax1, input_time, results.input_data[i, :], color=:blue)
    
    # Training Target
    ax2 = Axis(fig[i, 2])
    lines!(ax2, target_time, results.target_data[i, :], color=:green)
    
    # Open-loop Prediction
    ax3 = Axis(fig[i, 3])
    lines!(ax3, results.test_time, results.test_data[i, :], color=:black, linestyle=:dash, alpha=0.5, label="Target")
    lines!(ax3, results.test_time, results.predictive_output[i, :], color=:orange, label="Pred")
    
    # Closed-loop Prediction
    ax4 = Axis(fig[i, 4])
    lines!(ax4, results.test_time, results.test_data[i, :], color=:black, linestyle=:dash, alpha=0.5)
    lines!(ax4, results.test_time, results.closed_output[i, :], color=:red)
    
    # 마지막 행에만 xlabel 표시
    if i == N
        ax1.xlabel = "Time"
        ax2.xlabel = "Time"
        ax3.xlabel = "Time"
        ax4.xlabel = "Time"
    end
end

for i in 1:N
    rowsize!(fig.layout, i, Auto())
end

# 1부터 4까지의 모든 열이 동일한 비율로 확장되도록 설정
for j in 1:4
    colsize!(fig.layout, j, Auto())
end

# if @isdefined(data_used)
#     base_index = 1
#     closed_runs = build_closed_prediction_runs(data_used; shift=1, washout=1000, train_len=5000,
#                                                 predict_len=1250, rng=rng)
#     other_runs = closed_runs[setdiff(1:length(closed_runs), [base_index])]
#     fig_closed = plot_closed_predictions(closed_runs[base_index], other_runs)
#     display(fig_closed)
# end




# esn_LE = LyapunovExponent(esn, readout, 1; all_LE=false)[1]
# println(propertynames(esns[1]), ", ", propertynames(readouts[1]))

# fig3 = plot_prediction(rho_used, test_datas, outputs) 