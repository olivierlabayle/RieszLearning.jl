using RieszLearning
using Test

TESTDIR = joinpath(pkgdir(RieszLearning), "test")

@testset "RieszLearning.jl" begin
    @test include(joinpath(TESTDIR, "riesznet.jl"))
end
