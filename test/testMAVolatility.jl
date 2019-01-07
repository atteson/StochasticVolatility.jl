using StochasticVolatility
using Random
using StatsBase
using Distributions

Random.seed!(1)

model = MAVolatilityModel( 0.01, n -> 0.1 * 0.99^n )

n = 1000
(epsilon, delta, x) = rand( model, n )

logsigma = zeros(n)
for i = 1:n
    for j = 1:n
        logsigma[i] += model.f(j) * epsilon[n + i - j]
    end
end
z = x ./ (model.meanvol * exp.(logsigma))

@assert( abs(StatsBase.mean(z)*sqrt(length(z))) < 4 )
@assert( all(abs.([StatsBase.mean( z[1:end-d] .* z[d+1:end] ) * sqrt(length(z)-d) for d in 1:n-1]) .< 4) )

