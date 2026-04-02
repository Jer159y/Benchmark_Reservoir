using SparseArrays
using LinearAlgebra
using Arpack
using Random
using Statistics

# --- indexing_function_forward.m 변환 ---
function indexing_function_forward(chunk_end, locality, num_inputs)
    chunk_end = Int(chunk_end)
    locality = Int(locality)
    num_inputs = Int(num_inputs)
    
    if chunk_end + locality <= num_inputs
        return chunk_end+1 : chunk_end+locality
    elseif chunk_end + locality > num_inputs && chunk_end == num_inputs
        return 1 : mod(chunk_end + locality - 1, num_inputs) + 1
    elseif chunk_end + locality > num_inputs && chunk_end < num_inputs
        part1 = chunk_end+1 : num_inputs
        part2 = 1 : mod(chunk_end + locality - 1, num_inputs) + 1
        return vcat(part1, part2)
    else
        return Int[] # Error handling equivalent
    end
end

# --- indexing_function_rear.m 변환 ---
function indexing_function_rear(chunk_begin, locality, num_inputs)
    chunk_begin = Int(chunk_begin)
    locality = Int(locality)
    num_inputs = Int(num_inputs)
    
    if chunk_begin - locality > 0
        return chunk_begin-locality : chunk_begin-1
    elseif chunk_begin - locality <= 0 && chunk_begin > 1
        i1 = mod(chunk_begin - locality - 1, num_inputs) + 1
        return vcat(i1:num_inputs, 1:chunk_begin-1)
    elseif chunk_begin - locality <= 0 && chunk_begin == 1
        i1 = mod(chunk_begin - locality - 1, num_inputs) + 1
        return i1 : num_inputs
    else
        return Int[]
    end
end

# --- generate_reservoir.m 변환 ---
function generate_reservoir(size::Int, radius::Float64, degree::Int, labindex::Int, jobid::Int)
    # MATLAB: rng(labindex+jobid)
    Random.seed!(labindex + jobid)
    
    sparsity = degree / size
    A = sprand(size, size, sparsity)
    
    # eigs in Julia returns (vals, vectors)
    vals, _ = eigs(A; nev=1, which=:LM, ritzvec=false)
    e = maximum(abs.(vals))
    
    A = (A ./ e) .* radius
    return A
end

# --- reservoir_layer.m 변환 ---
function reservoir_layer(A, win, input, resparams)
    N = Int(resparams[:N])
    train_len = Int(resparams[:train_length])
    discard_len = Int(resparams[:discard_length])
    
    states = zeros(Float64, N, train_len)
    x = zeros(Float64, N)
    
    # Discard transient
    for i = 1:discard_len
        x = tanh.(A * x .+ win * input[:, i])
    end
    
    states[:, 1] = x
    
    # Collect states
    # MATLAB loop: for i = 1:train_length-1
    # states(:,i+1) = tanh(..., input(:, discard + i))
    for i = 1:train_len-1
        x = tanh.(A * states[:, i] .+ win * input[:, discard_len + i])
        states[:, i+1] = x
    end
    
    return states
end

# --- fit.m 변환 ---
function fit(params, states, data)
    beta = params[:beta]
    N = Int(params[:N])
    
    idenmat = beta * I(N)
    
    # MATLAB: w_out = data * states' * pinv(states * states' + idenmat)
    # Note: Using pinv for direct translation. For performance, considerations apply.
    w_out = data * states' * pinv(states * states' + idenmat)
    
    return w_out
end

# --- synchronize.m 변환 ---
function synchronize(W, x, w_in, data, prediction_marker, sync_length)
    curr_x = copy(x)
    for i = 1:sync_length
        curr_x = tanh.(W * curr_x .+ w_in * data[:, prediction_marker + i])
    end
    return curr_x
end