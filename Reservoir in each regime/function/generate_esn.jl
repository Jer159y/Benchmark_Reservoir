mutable struct HyperParams
    res_size::Int
    radius::Float64
    sparsity::Float64
    input_scaling::Float64
    leaky_coefficient::Float64
    washout::Int
end

function standardParam(X_train, ArgsRC)
    return Dict(
                :X_train => X_train,
                :in_size => size(X_train, 1),
                :res_size => ArgsRC.res_size,
                :reservoir => rand_sparse(; radius=ArgsRC.radius, sparsity=ArgsRC.sparsity),
                :input_layer => scaled_rand(; scaling=ArgsRC.input_scaling),
                :reservoir_driver => RNN(leaky_coefficient=ArgsRC.leaky_coefficient),
                :nla_type => NLADefault(),
                :washout => ArgsRC.washout,
                :states_type => StandardStates(),
                :initial_state => nothing # zeros(ArgsRC.res_size)
            )
end

function generate_esn(param, rng)
    esn = ESN(param[:X_train], param[:in_size], param[:res_size];
    reservoir=param[:reservoir], input_layer=param[:input_layer],
    reservoir_driver=param[:reservoir_driver],
    nla_type=param[:nla_type],
    washout=param[:washout],
    states_type=param[:states_type],
    rng=rng, initial_state=param[:initial_state])
    return esn
end

function copy_esn(esn, X_train, rng)
    esn = ESN(X_train, size(X_train, 1), esn.res_size;
        reservoir=(_, _, _, _) -> esn.reservoir_matrix,
        input_layer=(_, _, _, _) -> esn.input_matrix,
        reservoir_driver=esn.reservoir_driver,
        nla_type=esn.nla_type,
        washout=esn.washout,
        states_type=esn.states_type,
        rng=rng, initial_state=esn.states[:, 1])
    return esn
end