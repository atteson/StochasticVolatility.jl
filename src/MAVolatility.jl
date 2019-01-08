using FFTW

export MAVolatilityModel, simulate

struct MAVolatilityModel{F <: Function, T <: Real}
    meanvol::T
    f::F
end

function slowconvolve( u::AbstractVector{T},
                       v::AbstractVector{T} ) where {T <: Number}
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

function convolve!( u::AbstractVector{Complex{T}},
                    v::AbstractVector{Complex{T}},
                    w::AbstractVector{Complex{T}} ) where {T <: Real}
    fft!( u )
    fft!( v )
    w[1:length(w)] = u .* v
    ifft!( w )
end

function Base.rand( model::MAVolatilityModel{F}, n::Int ) where {F}
    m = 2*n-1
    N = 2^Int(ceil(log2(m))+1)
    epsilon = randn( m )
    delta = randn( n )
    epsilon2 = Complex{Float64}[epsilon; zeros(N - m)]
    a = Complex{Float64}[[model.f(i-1) for i in 1:n]; zeros(N - n)]
    convolve!( epsilon2, a, a )
    x = exp.(real.(a[n:m])) * model.meanvol .* delta
    return (epsilon, delta, x)
end
