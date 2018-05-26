__precompile__()

module InvariantCausal

    export causalSearch, screen_lasso, screen_HOLP, two_sample_chow, sukhatme_fisher_test, conditional_inv_test_chow, conditional_inv_test_logistic,
           GaussianSEM, simulate, causes, cov, random_gaussian_SEM, random_noise_intervened_SEM

    include("conditionalInvTests.jl")
    include("causalSearch.jl")
    include("SEM.jl")
    include("screening.jl")

    function _test_full()
        include(joinpath(@__DIR__, "..", "test", "test_full.jl"))
    end

end