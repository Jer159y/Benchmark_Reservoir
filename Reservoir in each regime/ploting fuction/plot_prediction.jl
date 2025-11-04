using CairoMakie
using LinearAlgebra

"""
before 2nd file.jl
"""

function plot_prediction(rho_used, test_datas, outputs)

    fig = Figure(size=(2000, 900), fontsize=14)
    axs = [Axis(fig[i, j]) for i in 1:3, j in 1:length(rho_used)]

    ts = 0.0:0.02:25.0
    lorenz_maxlyap = 0.9056
    lyap_time = (0:predict_len-1) .* 0.02 ./ lorenz_maxlyap
    coords = ["x(t)", "y(t)", "z(t)"]

    for (j, ρ) in enumerate(rho_used)
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

    return fig
end