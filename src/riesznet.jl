mutable struct RieszNetModel <: MLJBase.Deterministic
    lux_model
    hyper_parameters
end

RieszNetModel(;lux_model=MLP([2, 8]), hyper_parameters=()) = RieszNetModel(lux_model, hyper_parameters)

MLJBase.input_scitype(::Type{<:RieszNetModel}) = Tuple{MLJBase.Table(MLJBase.Continuous), Any}

function MLJBase.predict(lux_model::RieszNetModel, fitresult, X)
    T, W = X
    W_mat = permutedims(Matrix(W))
    T_hot = onehot_treatments(T)
    X_mat = vcat(T_hot, W_mat)
    lux_model, ps, st = fitresult
    ŷ, st = lux_model(X_mat, ps, st)
    return ŷ[1, :]
end

function MLJBase.fit(model::RieszNetModel, verbosity, X, indicators)
    return train_lux_model(model.lux_model, X..., indicators; 
        verbosity=verbosity,
        model.hyper_parameters...
    )
end

onehot_treatments(T) = mapreduce(vcat, eachcol(T)) do Tcol
    onehotbatch(Tcol, levels(Tcol))
end

function counterfactualTreatment(vals, Ts)
    n = nrows(Ts)
    counterfactual_Ts = map(enumerate(names(Ts))) do (i, T_name)
        T = Ts[!, T_name]
        categorical(fill(vals[i], n), 
            levels=levels(T), 
            ordered=isordered(T)
        )
    end
    DataFrame(counterfactual_Ts, names(Ts))
    return DataFrame(counterfactual_Ts, names(Ts))
end

struct RieszNetDataset
    W
    T
    T_cts
    function RieszNetDataset(T, W, indicators)
        W_mat = permutedims(Matrix(W))
        T_hot = onehot_treatments(T)
        T_cts = map(collect(indicators)) do (vals, sign)
            T_ct = counterfactualTreatment(vals, T)
            T_ct_hot = onehot_treatments(T_ct)
            (T_ct_hot, sign)
        end
        new(W_mat, T_hot, T_cts)
    end
end

MLUtils.numobs(dataset::RieszNetDataset) = size(dataset.W, 2)

function MLUtils.getobs(dataset::RieszNetDataset, idx)
    data_obs = vcat(dataset.T[:, idx], dataset.W[:, idx])
    data_cts = map(dataset.T_cts) do (T_ct, sign)
        vcat(T_ct[:, idx], dataset.W[:, idx]), sign
    end
    return data_obs, data_cts
end

function get_dataloaders(T, W, indicators;
    batch_size=32, 
    shuffle_before_split=true,
    train_ratio=0.7,
    loaders_parallel=false,
    rng=Random.default_rng()
    )
    dataset = RieszNetDataset(T, W, indicators)
    train_set, val_set = splitobs(rng, dataset, at=train_ratio, shuffle=shuffle_before_split)
    return (
        DataLoader(train_set, batchsize=batch_size, parallel=loaders_parallel),
        DataLoader(val_set, batchsize=batch_size, parallel=loaders_parallel)
    )
end

@doc raw"""
The Riesz Loss defined by:

```math
\frac{1}{n}\sum_{i=1}^n (\alpha(T, W)^2 - 2m(W, \alpha))
```

where:
- model: The Lux RieszNetModel
- ps: The model parameters
- st: The model state
- `alphas`: A tuple (alpha_obs, alpha_cts...) for observations and all counterfactuals
"""
function rieszloss(lux_model, ps, st, data)
    data_obs, data_cts = data
    # Observed part of the loss
    alpha_obs, st = lux_model(data_obs, ps, st)
    loss = alpha_obs.^2
    # Counterfactuals part of the loss
    for (data_ct, sign) in data_cts
        alpha_ct, st = lux_model(data_ct, ps, st)
        loss = loss .- 2 .* sign .* alpha_ct # Mutating not allowed by Zygote
    end

    stats = nothing
    return mean(loss), st, stats
end

function train_lux_model(lux_model, T, W, indicators; 
    dev=cpu_device(),
    batch_size=32,
    rng=Random.default_rng(),
    shuffle_before_split=true,
    backend = AutoZygote(),
    verbosity=1,
    optimiser=Adam(),
    nepochs=10,
    patience=5,
    train_ratio=0.7,
    loaders_parallel=false
    )
    # Get dataloaders
    train_dataloader, val_dataloader = dev(get_dataloaders(T, W, indicators;
            batch_size=batch_size, 
            shuffle_before_split=shuffle_before_split,
            train_ratio=train_ratio,
            loaders_parallel=loaders_parallel,
            rng=rng
        )
    )
    # Initialize model parameters and state
    ps, st = dev(Lux.setup(rng, lux_model))

    # Initialize the train state
    train_state = Training.TrainState(lux_model, ps, st, optimiser)

    # Training Loop
    train_losses = Vector{Union{Float64, Missing}}(undef, nepochs)
    val_losses = Vector{Union{Float64, Missing}}(undef, nepochs)
    best_parameters, best_states, best_val_loss, best_epoch = ps, st, Inf, 0
    for epoch in 1:nepochs
        # Train over dataset
        train_loss = 0.0
        for batch in train_dataloader
            gs, batch_train_loss, stats, train_state = Training.single_train_step!(
                backend, rieszloss, batch, train_state
            )
            train_loss += batch_train_loss
        end
        train_loss /= length(train_dataloader)
        train_losses[epoch] = train_loss
        # Evaluate on validation set
        st_ = Lux.testmode(train_state.states)
        val_loss = 0.0
        for batch in val_dataloader
            batch_val_loss, _, _ = rieszloss(lux_model, train_state.parameters, st_, batch)
            val_loss += batch_val_loss
        end
        val_loss /= length(val_dataloader)
        val_losses[epoch] = val_loss
        # Logging
        verbosity > 0 && @info @sprintf("Epoch [%s] - Train Loss: %.5f, Val Loss: %.5f\n", epoch, train_loss, val_loss)
        # Check for early stopping
        if val_loss >= best_val_loss && epoch - best_epoch >= patience
            verbosity > 0 && @info @sprintf("Early stopping at epoch %s", epoch)
            break
        else
            if val_loss < best_val_loss
                best_parameters, best_states, best_val_loss, best_epoch = 
                    deepcopy(train_state.parameters), deepcopy(train_state.states), val_loss, epoch
            end
        end 
    end

    # Make MLJBase compatible output
    fitresult = (lux_model, best_parameters, Lux.testmode(best_states))
    cache = nothing
    report = (train_losses=train_losses, val_losses=val_losses, best_epoch=best_epoch)
    return fitresult, cache, report
end

"""
    MLP(sizes; activation=relu)

Builds a simple MLP model with the given hidden sizes. The last layer has a single output which must not be given.

Example usage:

```julia
model = MLP([9, 32, 8]; activation=relu)
```

The model has an input size of 9, two hidden layers with sizes 32 and 8, and a single output layer.

"""
function MLP(sizes; activation=relu)
    hidden_layers = [Dense(sizes[i], sizes[i+1], activation) for i in 1:length(sizes)-1]
    Chain(
        hidden_layers...,
        Dense(sizes[end], 1)
    )
end