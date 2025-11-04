W_out1 = readouts[1].output_matrix
W_out2 = readouts[2].output_matrix
W_out3 = readouts[3].output_matrix

last_value1 = readouts[1].last_value
last_value2 = readouts[2].last_value
last_value3 = readouts[3].last_value

esn1 = esns[1]
esn2 = esns[2]
esn3 = esns[3]

c = 0.0:0.1:1.0
outputs_c = []
output_layer_new = deepcopy(readouts[1])

for (i, c_val) in enumerate(c)
    W_out_new = (1 - c_val) * W_out2 + c_val * W_out3
    output_layer_new.output_matrix[:,:] = W_out_new
    # output_layer_new.last_value[:] = last_value1
    output = esn1(Generative(predict_len), output_layer_new)
    push!(outputs_c, output)
end


GLMakie.activate!() # (title = "Custom title", fxaa = false)

fig = Figure(size = (2000, 1200), fontsize = 12)

left_grid = fig[1, 1] = GridLayout()
l_axs = [Axis(left_grid[i, 1]) for i in 1:3]
l_axs[1].title = "Trajectory for each axis"

right_grid = fig[1, 2] = GridLayout()
ax3d = Axis3(right_grid[1, 1],
             title = "Interpolated Phase Space",
             xlabel = "x", ylabel = "y", zlabel = "z")

bottom_grid = fig[2, 1:2] = GridLayout()
bl_axs = [Axis(bottom_grid[1, j]) for j in 1:3]
bl_axs[2].title = "Phase Space Projections"


ts = 0.0:0.02:25.0
coords = ["x(t)", "y(t)", "z(t)"]
phases = ["x-y", "y-z", "z-x"]
colors = [get(ColorSchemes.viridis, i / length(c)) for i in 1:length(c)]
lines = []

for i in 1:3
    ax = l_axs[i]
    for (c_index, c_val) in enumerate(c)
        output = outputs_c[c_index]
        l = lines!(ax, ts[2:end], output[i, :], color = colors[c_index], alpha = 0.8,
                linewidth = 1.5, label = "c = $(round(c_val, digits=2))")
        push!(lines, l)
    end
    ax.ylabel = coords[i]
end
l_axs[3].xlabel = "t"

for j in 1:3
    ax = bl_axs[j]
    for (c_index, c_val) in enumerate(c)
        output = outputs_c[c_index]
        x = output[1, :]
        y = output[2, :]
        z = output[3, :]
        if j == 1
            l = lines!(ax, x, y, color=colors[c_index], alpha = 0.8,
                   linewidth = 1.5, label = "c = $(round(c_val, digits=2))")
        elseif j == 2
            l = lines!(ax, y, z, color=colors[c_index], alpha = 0.8,
                   linewidth = 1.5, label = "c = $(round(c_val, digits=2))")
        else
            l = lines!(ax, z, x, color=colors[c_index], alpha = 0.8,
                   linewidth = 1.5, label = "c = $(round(c_val, digits=2))")
        end
        push!(lines, l)
   end
    ax.xlabel = coords[j]
    ax.ylabel = j+1 < 4 ? coords[j+1] : coords[1]
end

for (c_index, c_val) in enumerate(c)
    output = outputs_c[c_index]
    x = output[1, :]
    y = output[2, :]
    z = output[3, :]
    l = lines!(ax3d, x, y, z; color=colors[c_index], linewidth = 1.5, transparency = true,
                   alpha=0.8, label = "c = $(round(c_val, digits = 2))")
   push!(lines, l)
end

grouped = Dict{String, Vector{Lines}}()
for l in lines
    label = l.label[]
    grouped[label] = get(grouped, label, Lines[])  # 없으면 빈 벡터 생성
    push!(grouped[label], l)
end

sorted_labels = sort(collect(keys(grouped)))
sorted_values = [grouped[label] for label in sorted_labels]

legend = Legend(bottom_grid[1, 4], sorted_values, sorted_labels; title = "c values", position = :lt)
rowsize!(fig.layout, 1, Relative(0.6))

save("interpolation_rho28_rho100.png", fig)