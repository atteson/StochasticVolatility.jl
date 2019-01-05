using ToeplitzMatrices
using Random
using LinearAlgebra
using PyPlot

Random.seed!(1)

sizes = [10,20,50,100,200,500,1000,2000,5000,10000,20000,50000]
times = Float64[]
for n in sizes
    r = rand(n)
    c = [r[1];rand(n-1)]
    T = Toeplitz( c, r )

    t0 = time()
    det(T)
    t1 = time()
    push!( times, t1 - t0 )
end

times ./ sizes

times ./ (sizes.^2)
