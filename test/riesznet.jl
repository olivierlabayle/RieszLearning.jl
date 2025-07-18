module TestRieszNet

using RieszLearning
using Test
using MLJBase
using Random
using Distributions
using StableRNGs
using StatsBase
using DataFrames
using MLUtils
using Lux
using Optimisers

function evaluate_true_riesz_representer(T, Tprobs, encoding; case=("CC", "AA"), control=("CG", "AT"))
    linear_indices = LinearIndices(encoding)
    case_index = linear_indices[findfirst(==(case), encoding)]
    control_index = linear_indices[findfirst(==(control), encoding)]
    riesz_true = map(enumerate(eachrow(T))) do (i, Trow)
        Trow_vals = Tuple(Trow)
        if Trow_vals == case
             1 / Tprobs[i, case_index]
        elseif Trow_vals == control
            -1 / Tprobs[i, control_index]
        else
            0
        end
    end
    return riesz_true
end

function binary_outcome_categorical_treatment_pb(;n=100)
    rng = StableRNG(123)
    # Sampling W:
    W = float(rand(rng, Bernoulli(0.5), n, 3))
    # Sampling T from W:
    # T₁, T₂ will have 3 categories each
    # This is embodied by a 9 dimensional full joint
    θ = rand(rng, 3, 9)
    Tprobs = exp.(W*θ) ./ sum(exp.(W*θ), dims=2)
    encoding = collect(Iterators.product(["CC", "GG", "CG"], ["TT", "AA", "AT"]))
    T = [sample(rng, encoding, Weights(Tprobs[i, :])) for i in 1:n]
    T = (T₁=categorical([t[1] for t in T]), T₂=categorical([t[2] for t in T]))

    dataset = DataFrame(T₁=T.T₁, T₂=T.T₂, W₁=W[:, 1], W₂=W[:, 2], W₃=W[:, 3])

    return dataset, Tprobs, encoding
end

@testset "Test RieszNetDataset and get_dataloaders" begin
    raw_dataset, _, _ = binary_outcome_categorical_treatment_pb(;n=100)
    # This preparation would be done by TMLE.jl
    T = raw_dataset[!, [:T₁, :T₂]]
    W = raw_dataset[!, [:W₁, :W₂, :W₃]]
    indicators = Dict(("CC", "AA") => 1, ("CG", "AT") => -1)
    # Test accessors
    dataset = RieszLearning.RieszNetDataset(T, W, indicators)
    @test numobs(dataset) == 100
    # An observation consists in both observed and counterfactual data
    obs_1 = getobs(dataset, 1:2)
    obs_1_obs, obs_1_cts = obs_1
    @test obs_1_obs == [
                 # Sample 1 Sample 2 
        1.0  1.0 # CC CC
        0.0  0.0 # CG CG
        0.0  0.0 # GG GG
        1.0  0.0 # AA AA
        0.0  1.0 # AT AT
        0.0  0.0 # TT TT
        1.0  1.0 # W₁ W₁
        0.0  0.0 # W₂ W₂
        0.0  1.0 # W₃ W₃
    ]
    sort!(obs_1_cts, by=x -> x[2])
    obs_1_ct, indicator_sign = obs_1_cts[1]
    @test indicator_sign == -1
    @test obs_1_ct == [
        0.0  0.0
        1.0  1.0 # CG CG
        0.0  0.0
        0.0  0.0
        1.0  1.0 # AT AT
        0.0  0.0
        1.0  1.0 # Ws unchanged
        0.0  0.0
        0.0  1.0
    ]
    obs_1_ct, indicator_sign = obs_1_cts[2]
    @test indicator_sign == 1
    @test obs_1_ct == [
        1.0  1.0 # CC CC
        0.0  0.0 
        0.0  0.0
        1.0  1.0 # AA AA
        0.0  0.0 
        0.0  0.0
        1.0  1.0 # Ws unchanged
        0.0  0.0
        0.0  1.0
    ]

    train_dataloader, val_dataloader = RieszLearning.get_dataloaders(T, W, indicators; 
        batch_size=10, 
        train_ratio=0.8, 
        shuffle_before_split=true
    )
    @test length(train_dataloader) == 8 # 80% of 100 is 80, batch size is 10
    @test length(val_dataloader) == 2 # 20% of 100 is 20
    @test first(train_dataloader)[1] !== getobs(dataset, 1:10)[1] # shuffled
end

@testset "Test RieszNetModel" begin
    raw_dataset, Tprobs, encoding = binary_outcome_categorical_treatment_pb(;n=10_000)
    # This preparation would be done by TMLE.jl
    T = raw_dataset[!, [:T₁, :T₂]]
    W = raw_dataset[!, [:W₁, :W₂, :W₃]]
    indicators = Dict(("CC", "AA") => 1, ("CG", "AT") => -1)
    train_dataloader, val_dataloader = RieszLearning.get_dataloaders(T, W, indicators; 
        batch_size=10, 
        train_ratio=0.8, 
        shuffle_before_split=true
    )
    
    # Define a simple RieszNetModel
    model = RieszNetModel(
        lux_model=Chain(
            Dense(9, 32, relu),
            Dense(32, 8, relu),
            Dense(8, 1)
        ),
        hyper_parameters=(
            nepochs=5,
            rng=StableRNG(123),
            optimiser=Adam(0.01)
        )
    )
    mach = machine(model, (T, W), indicators)
    # Test fit method
    fit!(mach, verbosity=0)
    train_losses, val_losses, best_epoch = report(mach)
    @test minimum(skipmissing(val_losses)) == val_losses[best_epoch]
    # Check if we get closer to the true function with more epochs
    y = evaluate_true_riesz_representer(T, Tprobs, encoding; case=("CC", "AA"), control=("CG", "AT"))
    ## First evaluation after 5 epochs
    ŷ = MLJBase.predict(mach, (T, W))
    mse_5_epochs = mean((ŷ .- y).^2)
    ## Second evaluation after 25 epochs
    model.hyper_parameters = (nepochs=20, rng=StableRNG(123))
    fit!(mach, verbosity=0)
    ŷ = MLJBase.predict(mach, (T, W))
    mse_25_epochs = mean((ŷ .- y).^2)
    @test mse_25_epochs < mse_5_epochs
    ## Second evaluation after 125 epochs
    model.hyper_parameters = (nepochs=100, rng=StableRNG(123))
    fit!(mach, verbosity=0)
    ŷ = MLJBase.predict(mach, (T, W))
    mse_125_epochs = mean((ŷ .- y).^2)
    @test mse_125_epochs < mse_25_epochs
end

end

true