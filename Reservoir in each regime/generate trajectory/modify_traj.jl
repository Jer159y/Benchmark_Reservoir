# Modification functions

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


# Data modification utilities (multiple data sets)

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

function apply_modification(data_used, modify_function)
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

# Example

# decay_data = apply_modification(data_used, (ρ, u0, t, u) -> (ρ, u0, t, exponential_decay(u, t, 0.01)))
# cutting_decay = cutting(decay_data, 5000)
# check_data_plot(8, data_used, decay_data)
