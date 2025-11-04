using GLMakie, CairoMakie

"""
before 1st file.jl and 2nd file.jl
"""

function plot_bifurcation_diagram(rho_vals_x, x_max_vals, rho_vals_z, z_max_vals, data_used)
    println("\nPlotting results...")
    fig = Figure(size=(1500, 1500), fontsize=14)

    ax1 = Axis(fig[1, 1],
            xlabel = "ρ",
            ylabel = "x maxima",
            title = "Lorenz System Bifurcation Diagram")
            
    ax2 = Axis(fig[2, 1],
            xlabel = "ρ",
            ylabel = "z maxima",
            title = "Lorenz System Bifurcation Diagram")

    Makie.scatter!(ax1, rho_vals_x, x_max_vals,
                            markersize = 0.5,
                            color = :black)
    vlines!(ax1, [23.7], color=:red, label="ρ=23.7")
    vlines!(ax1, [23.0, 28.0, 100.0], color=:blue, label="ρ used in reservoir")

    Makie.scatter!(ax2, rho_vals_z, z_max_vals,
                            markersize = 0.5,
                            color = :black)
    vlines!(ax2, [23.7], color=:red, label="ρ=23.7")
    vlines!(ax2, [23.0, 28.0, 100.0], color=:blue, label="ρ used in reservoir")

    # save("lorenz_bifurcation_diagram.png", fig)
    # GLMakie.closeall()
    return fig
end

function plot_phase_space(data_used)
    println("\nPlotting phase space trajectories...")

    n = length(data_used)
    fig_phase = Figure(size=(1200, 320*n), fontsize=12)

    for (i, d) in enumerate(data_used)
        x = d.u[1, :]
        y = d.u[2, :]
        z = d.u[3, :]

        ax_xy = Axis(fig_phase[i, 1], title = "ρ = $(d.ρ) : x-y", xlabel = "x", ylabel = "y")
        lines!(ax_xy, x, y, color = :navy, linewidth = 0.6)

        ax_yz = Axis(fig_phase[i, 2], title = "ρ = $(d.ρ) : y-z", xlabel = "y", ylabel = "z")
        lines!(ax_yz, y, z, color = :green, linewidth = 0.6)

        ax_zx = Axis(fig_phase[i, 3], title = "ρ = $(d.ρ) : z-x", xlabel = "z", ylabel = "x")
        lines!(ax_zx, z, x, color = :firebrick, linewidth = 0.6)
    end

    # save("lorenz_phase_space.png", fig_phase)
    return fig_phase
end