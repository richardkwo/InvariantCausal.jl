__precompile__()

module InvariantCausal

    export causalSearch, screen_lasso, two_sample_chow, conditional_inv_test_chow, conditional_inv_test_logistic,
           GaussianSEM, simulate, causes, cov, random_gaussian_SEM, random_noise_intervened_SEM

    include("conditionalInvTests.jl")
    include("causalSearch.jl")
    include("SEM.jl")
    include("screening.jl")

    function _test_full()
        include(joinpath(@__DIR__, "..", "test", "test_full.jl"))
    end

end