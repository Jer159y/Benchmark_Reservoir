function create_states(reservoir_driver::AbstractReservoirDriver,
        train_data::AbstractArray, washout::Int, reservoir_matrix::AbstractMatrix,
        input_matrix::AbstractMatrix, bias_vector::AbstractArray,
        initial_state::AbstractArray)

    train_len = size(train_data, 2) - washout
    res_size = size(reservoir_matrix, 1)
    states = adapt(typeof(train_data), zeros(res_size, train_len))
    tmp_array = allocate_tmp(reservoir_driver, typeof(train_data), res_size)
    _state = adapt(typeof(train_data), reshape(initial_state, res_size, 1))

    for i in 1:washout
        yv = @view train_data[:, i]
        _state = next_state!(_state, reservoir_driver, _state, yv, reservoir_matrix,
            input_matrix, bias_vector, tmp_array)
    end

    for j in 1:train_len
        yv = @view train_data[:, washout + j]
        _state = next_state!(_state, reservoir_driver, _state, yv,
            reservoir_matrix, input_matrix, bias_vector, tmp_array)
        states[:, j] = _state
    end

    return states
end

function create_states(reservoir_driver::AbstractReservoirDriver,
        train_data::AbstractArray, washout::Int, reservoir_matrix::Vector,
        input_matrix::AbstractArray, bias_vector::AbstractArray,
        initial_state::AbstractArray)
    train_len = size(train_data, 2) - washout
    res_size = sum([size(reservoir_matrix[i], 1) for i in 1:length(reservoir_matrix)])
    states = adapt(typeof(train_data), zeros(res_size, train_len))
    tmp_array = allocate_tmp(reservoir_driver, typeof(train_data), res_size)
    _state = adapt(typeof(train_data), reshape(initial_state, res_size, 1))

    for i in 1:washout
        for j in 1:length(reservoir_matrix)
            _inter_state = next_state!(_inter_state, reservoir_driver, _inter_state,
                train_data[:, i],
                reservoir_matrix, input_matrix, bias_vector,
                tmp_array)
        end
        _state = next_state!(_state, reservoir_driver, _state, train_data[:, i],
            reservoir_matrix, input_matrix, bias_vector, tmp_array)
    end

    for j in 1:train_len
        _state = next_state!(_state, reservoir_driver, _state, train_data[:, washout + j],
            reservoir_matrix, input_matrix, bias_vector, tmp_array)
        states[:, j] = _state
    end

    return states
end

function ESN(train_data::AbstractArray, in_size::Int, res_size::Int;
        input_layer = scaled_rand, reservoir = rand_sparse, bias = zeros32,
        reservoir_driver::AbstractDriver = RNN(),
        nla_type::NonLinearAlgorithm = NLADefault(),
        states_type::AbstractStates = StandardStates(),
        washout::Int = 0, rng::AbstractRNG = Utils.default_rng(),
        initial_state::AbstractArray = nothing,
        matrix_type = typeof(train_data))
    if states_type isa AbstractPaddedStates
        in_size = size(train_data, 1) + 1
        train_data = vcat(adapt(matrix_type, ones(1, size(train_data, 2))),
            train_data)
    end

    T = eltype(train_data)
    reservoir_matrix = reservoir(rng, T, res_size, res_size)
    input_matrix = input_layer(rng, T, res_size, in_size)
    bias_vector = bias(rng, res_size)
    inner_res_driver = reservoir_driver_params(reservoir_driver, res_size, in_size)
    if initial_state === nothing
        states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector)
    else
        states = create_states(inner_res_driver, train_data, washout, reservoir_matrix,
        input_matrix, bias_vector, initial_state)
    end
    
    train_data = train_data[:, (washout + 1):end]

    return ReservoirComputing.ESN(res_size, train_data, nla_type, input_matrix,
        inner_res_driver, reservoir_matrix, bias_vector, states_type, washout,
        states)
end