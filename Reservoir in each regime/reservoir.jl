"""
2nd file
"""

function exponential_decay(u, t, decay_rate)
    decay_factors = exp.(-decay_rate .* t)
    decay_factors = repeat(decay_factors', size(u, 1), 1)
    u_modified = u .* decay_factors
    return u_modified
end

function transient_removal(u, t, transient_time)
    indices = findall(t .>= transient_time)
    println("Removing transient up to t = $transient_time. New data length: $(length(indices))")
    t_new = t[indices]
    u_new = u[:, indices]
    return t_new, u_new
end

function cutting(datas, cut_len)
    new_datas = []
    u_len = size(datas[1].u, 2)
    num = floor(Int, u_len / cut_len)
    println("Generate $num segments.")
    for data in datas
        for i in 0:(num - 1)
            start_idx = i * cut_len + 1
            end_idx = start_idx + cut_len - 1
            t_segment = data.t[start_idx:end_idx]
            u_segment = data.u[:, start_idx:end_idx]
            push!(new_datas, (ρ=data.ρ, u0=data.u0, t=t_segment, u=u_segment))
        end
    end
    return new_datas
end

function modify_data(data_used, modify_function)
    new_data = []
    for data in data_used
        ρ = data.ρ
        u0 = data.u0
        t = data.t
        u = data.u

        ρ_new, u0_new, t_new, u_new = modify_function(ρ, u0, t, u)
        push!(new_data, (ρ=ρ_new, u0=u0_new, t=t_new, u=u_new))
    end
    return new_data
end

function check_data_plot(num, data_used, new_data)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
            xlabel = "Time",
            ylabel = "State Variables",
            title = "Lorenz System State Variables Over Time (Data Set $num)")
    lines!(ax, data_used[num].t, data_used[num].u[1, :], color = :blue, alpha=0.3, label = "x(t)")
    lines!(ax, new_data[num].t, new_data[num].u[1, :], color = :red, alpha=0.3, label = "Modified x(t)")
    display(fig)
end

function check_matrix_spectral(esns, readouts)
    for i in 1:length(esns)
        esn = esns[i]
        readout = readouts[i]
        W_in = esn.input_matrix
        W_res = esn.reservoir_matrix
        W_out = readout.output_matrix
        W = W_res + W_in * W_out
        ρ_res = maximum(abs.(eigvals(W_res)))
        ρ_W = maximum(abs.(eigvals(W)))
        println("ρ = $(round(ρ_res, digits=4)), ρ(W) = $(round(ρ_W, digits=4))") # 고민 좀...
    end
end

function train_test_split(data, shift, washout, train_len, predict_len)
    input_data = data.u[:, shift:(shift + train_len - 1)]
    target_data = data.u[:, (washout + shift + 1):(shift + train_len)]
    test_data = data.u[:, (shift + train_len + 1):(shift + train_len + predict_len)]
    return input_data, target_data, test_data
end

function gen_trained_data(data; shift=1, washout=1000, train_len=5000, predict_len=1250, rng=MersenneTwister(42))

    args = HyperParams(1500, 0.1, 30 / 1500, 0.1, 0.2, washout)
    training_method = StandardRidge(1e-6)

    input_data, target_data, test_data = train_test_split(data, shift, washout, train_len, predict_len)
    esn_param = standardParam(input_data, args)
    esn_param[:initial_state] = zeros(args.res_size)
    esn = generate_esn(esn_param, rng)
    output_layer = train(esn, target_data, training_method)
    
    return (input_data=input_data, target_data=target_data, test_data=test_data,
            esn=esn, readout=output_layer, param=esn_param)
end

function run_closed_prediction(data; shift=1, washout=1000, train_len=5000, predict_len=1250, rng=MersenneTwister(42))
    args = HyperParams(1500, 0.1, 30 / 1500, 0.1, 0.2, washout)
    training_method = StandardRidge(1e-6)

    input_data, target_data, test_data = train_test_split(data, shift, washout, train_len, predict_len)

    esn_param = standardParam(input_data, args)
    esn_param[:initial_state] = zeros(args.res_size)
    esn = generate_esn(esn_param, rng)
    readout = train(esn, target_data, training_method)

    predictive_output = esn(Predictive(test_data), readout)
    closed_output = esn(Generative(predict_len), readout)

    test_start = shift + train_len + 1
    test_range = test_start:(test_start + predict_len - 1)
    test_time = data.t[test_range]

    return (ρ=data.ρ, u0=data.u0, input_data=input_data, target_data=target_data, test_data=test_data,
            test_time=test_time, predictive_output=predictive_output, closed_output=closed_output,
            esn=esn, readout=readout, param=esn_param)
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


rng = rand(Int); println("Random seed: ", rng)
rng = MersenneTwister(rng)

cutting_data = cutting(data_used, 5000)

decay_data = modify_data(data_used, (ρ, u0, t, u) -> (ρ, u0, t, exponential_decay(u, t, 0.01)))
cutting_decay = cutting(decay_data, 5000)
check_data_plot(8, data_used, decay_data)

transient_data = modify_data(data_used, (ρ, u0, t, u) -> (ρ, u0, transient_removal(u, t, 50.0)...))
check_data_plot(8, data_used, transient_data)

results0 = gen_trained_data(data_used[1]; rng=rng)
results1 = gen_trained_data(decay_data[1]; rng=rng)
results2 = gen_trained_data(transient_data[1]; rng=rng)
results3 = gen_trained_data(cutting_decay[1]; rng=rng)

if @isdefined(data_used)
    base_index = 1
    closed_runs = build_closed_prediction_runs(data_used; shift=1, washout=1000, train_len=5000,
                                                predict_len=1250, rng=rng)
    other_runs = closed_runs[setdiff(1:length(closed_runs), [base_index])]
    fig_closed = plot_closed_predictions(closed_runs[base_index], other_runs)
    display(fig_closed)
end




# esn_LE = LyapunovExponent(esn, readout, 1; all_LE=false)[1]
# println(propertynames(esns[1]), ", ", propertynames(readouts[1]))

# fig3 = plot_prediction(rho_used, test_datas, outputs)