using FFTW

export MAVolatilityModel, simulate

struct MAVolatilityModel{F <: Function, T <: Real}
    meanvol::T
    f::F
end

function convolve!( u::AbstractVector{Complex{T}},
                    v::AbstractVector{Complex{T}},
                    w::AbstractVector{Complex{T}} ) where {T <: Number}
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
    x = exp.(real.(a[1:n])) * model.meanvol .* delta
    return (epsilon, delta, x)
end
