using StochasticVolatility
using Random
using StatsBase
using Distributions

Random.seed!(1)

model = MAVolatilityModel( 0.01, n -> 0.1 * 0.99^n )

n = 1000
(epsilon, delta, x) = rand( model, n )

a = [model.f(i) for i in 0:n-1]

m = 2 * n - 1
N = 2^Int(ceil(log2(m))+1)
epsilon2 = Complex{Float64}[epsilon; zeros(N - m)]
a2 = Complex{Float64}[a; zeros(N - n)]
logsigma = zeros(Complex{Float64}, N)
StochasticVolatility.convolve!( epsilon2, a2, logsigma )
StochasticVolatility.convolve!( epsilon2, a2, a2 )
@assert( maximum(abs.(logsigma[n:2n-1] - StochasticVolatility.slowconvolve( a, epsilon ))) < 1e-12 )

z = x ./ (model.meanvol * exp.(real.(logsigma[n:m])))
maximum(abs.(z - delta))
