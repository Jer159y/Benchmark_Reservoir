"""
2nd file
"""

shift = 300
washout = 1000
train_len = 5000
predict_len = 1250

test_datas = []
outputs = []
esns = []
readouts = []

args = HyperParams(1500, 1.2, 30 / 1500, 0.1, 0.2, washout)
training_method = StandardRidge(1e-6)

rng = rand(Int); println("Random seed: ", rng)
rng = MersenneTwister(rng)

for (i, data) in enumerate(data_used)
    input_data = data.u[:, shift:(shift + train_len - 1)]
    target_data = data.u[:, (washout + shift + 1):(shift + train_len)]
    test_data = data.u[:, (shift + train_len + 1):(shift + train_len + predict_len)]
    push!(test_datas, test_data)

    if i == 1
        println("Training ESN for ρ = $(rho_used[i])")
        esn_param = standardParam(input_data, args)
        esn_param[:initial_state] = zeros(args.res_size)
        esn = generate_esn(esn_param, rng)
    else
        println("Training ESN for ρ = $(rho_used[i]) (copying previous ESN)")
        esn = copy_esn(esns[i-1], input_data, rng)
    end
    push!(esns, esn)
    output_layer = train(esn, target_data, training_method)
    output = esn(Generative(predict_len), output_layer)
    push!(outputs, output)
    push!(readouts, output_layer)
end

# println(propertynames(esns[1]), ", ", propertynames(readouts[1]))

for i in 1:length(esns)
    esn = esns[i]
    readout = readouts[i]
    W_in = esn.input_matrix
    W_res = esn.reservoir_matrix
    W_out = readout.output_matrix
    W = W_res + W_in * W_out
    ρ_res = maximum(abs.(eigvals(W_res)))
    ρ_W = maximum(abs.(eigvals(W)))
    println("ρ = $(rho_used[i]): ρ = $(round(ρ_res, digits=4)), ρ(W) = $(round(ρ_W, digits=4))")
end

# esn_LE = LyapunovExponent(esn, readout, 1; all_LE=false)[1]
# println(propertynames(esns[1]), ", ", propertynames(readouts[1]))

# fig3 = plot_prediction(rho_used, test_datas, outputs)