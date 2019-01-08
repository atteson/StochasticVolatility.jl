using FFTW
using Distributions

export MAVolatilityModel, simulate

struct MAVolatilityModel{F <: Function, T <: Real}
    meanvol::T
    f::F
    cutoff::Int
end

function slowconvolve(
    u::AbstractVector{T},
    v::AbstractVector{T},
) where {T <: Number}
    n = length(u)
    @assert( length(v) == 2*n-1 )
    w = zeros(min(length(u),length(v)))
    for i = 1:n
        for j = 1:n
            w[i] += u[j] * v[n + i - j]
        end
    end
    return w
end

function convolve!(
    u::AbstractVector{Complex{T}},
    v::AbstractVector{Complex{T}},
    w::AbstractVector{Complex{T}},
) where {T <: Real}
    fft!( u )
    fft!( v )
    w[1:length(w)] = u .* v
    ifft!( w )
end

function getsigma(
    model::MAVolatilityModel{F},
    epsilon::AbstractVector{T},
) where {F, T <: Real}
    c = model.cutoff
    m = length(epsilon)
    n = m - c + 1
    N = 2^Int(ceil(log2(m))+1)
    epsilon2 = Complex{Float64}[epsilon; zeros(N - m)]
    a = Complex{Float64}[[model.f(i-1) for i in 1:c]; zeros(N - c)]
    convolve!( epsilon2, a, a )
    return exp.(real.(a[n:m])) * model.meanvol
end

function Base.rand(
    model::MAVolatilityModel{F},
    n::Int,
) where {F}
    c = model.cutoff
    m = c + n - 1
    epsilon = randn( m )
    delta = randn( n )

    sigma = getsigma( model, epsilon )
    
    x = sigma .* delta
    return (epsilon, delta, x)
end

function loglikelihood(
    model::MAVolatilityModel{F},
    epsilon::AbstractVector{T},
    lo::AbstractVector{T},
    hi::AbstractVector{T},
) where {F, T <: Real}
    n = Normal()
    ll = loglikelihood( n, epsilon )
    sigma = getsigma( model, epsilon )
    for i = 1:length(lo)
        ll *= cdf( n, hi[i]/sigma[i] ) - cdf( n, lo[i]/sigma[i] )
    end
    return ll
end
                        
