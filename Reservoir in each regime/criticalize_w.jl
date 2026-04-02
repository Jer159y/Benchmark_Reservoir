function check_welltrain_plot(time_axis, welltrained_test_data, welltrained_output_pred, welltrained_output_gen)
    error1 = sum(welltrained_output_pred .- welltrained_test_data, dims=1)
    error2 = sum(welltrained_output_gen .- welltrained_test_data, dims=1)
    fig = Figure(size=(800, 600))
    ax = Axis(fig[1, 1],
            ylabel = "X",
            title = "Lorenz System State Variables Over Time (Well-Trained ESN)")
    lines!(ax, time_axis, welltrained_output_pred[1, :], color = :blue, alpha=0.3, label = "ESN Output_Predictive")
    lines!(ax, time_axis, welltrained_test_data[1, :], color = :red, alpha=0.3, label = "Target Data")
    lines!(ax, time_axis, welltrained_output_gen[1, :], color = :orange, alpha=0.3, label = "ESN Output_Generative")
    ax = Axis(fig[2, 1],
            ylabel = "Y")
    lines!(ax, time_axis, welltrained_output_pred[2, :], color = :blue, alpha=0.3, label = "ESN Output_Predictive")
    lines!(ax, time_axis, welltrained_test_data[2, :], color = :red, alpha=0.3, label = "Target Data")
    lines!(ax, time_axis, welltrained_output_gen[2, :], color = :orange, alpha=0.3, label = "ESN Output_Generative")
    ax = Axis(fig[3, 1],
            xlabel = "Time",
            ylabel = "Z")
    lines!(ax, time_axis, welltrained_output_pred[3, :], color = :blue, alpha=0.3, label = "ESN Output_Predictive")
    lines!(ax, time_axis, welltrained_test_data[3, :], color = :red, alpha=0.3, label = "Target Data")
    lines!(ax, time_axis, welltrained_output_gen[3, :], color = :orange, alpha=0.3, label = "ESN Output_Generative")
    Legend(fig[1, 2], ax; position = :rt)
    ax = Axis(fig[2, 2],
            xlabel = "Time",  
            ylabel = "Error",
            title = "Prediction Errors Over Time")
    lines!(ax, time_axis, error1[1, :], color = :green, alpha=0.3, label = "Error (Pred)")
    lines!(ax, time_axis, error2[1, :], color = :purple, alpha=0.3, label = "Error (Gen)")
    Legend(fig[3, 2], ax; position = :rt)
    display(fig)
end

function criticalize_w(param, esn, readout, training_method, rng,
                        input_datas, target_datas, test_datas, cycle)
    predict_len = size(test_datas[1], 2)
    for num in cycle
        println("Loop $num / $(length(cycle))")
        W_in = esn.input_matrix
        W_res = esn.reservoir_matrix
        W_out = readout.output_matrix
        W_new = W_res + W_in * W_out

        param[:X_train] = input_datas[num]
        param[:reservoir] = (_, _, _, _) -> W_new
        param[:input_layer] = (_, _, _, _) -> W_in

        esn = generate_esn(param, rng)

        readout = train(esn, target_datas[num], training_method)
        output1 = esn(Predictive(test_datas[num]), readout)
        output2 = esn(Generative(predict_len), readout)

        check_welltrain_plot(1:predict_len, test_datas[num], output1, output2)

        ρ_res = maximum(abs.(eigvals(W_res)))
        ρ_W = maximum(abs.(eigvals(W_new)))
        println("before ρ(W) = $(round(ρ_res, digits=4)), after ρ(W) = $(round(ρ_W, digits=4))")
    end
end

cycle = 1:9
segs = decay_data[cycle.+1]
triples = [train_test_split(d, 1, 1000, 5000, 1250) for d in segs]
input_datas, target_datas, test_datas = (getindex.(triples, i) for i in 1:3)

param = deepcopy(results1.param)
esn = deepcopy(results1.esn)
readout = deepcopy(results1.readout)

criticalize_w(param, esn, readout, StandardRidge(1e-2), rng,
                input_datas, target_datas, test_datas, cycle)

