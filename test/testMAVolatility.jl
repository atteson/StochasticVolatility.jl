using StochasticVolatility

model = MAVolatilityModel( n -> 1e-4 * 0.99^n )

(epsilon, x) = rand( model, 100 )

