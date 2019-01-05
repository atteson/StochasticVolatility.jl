using FFTW

export MAVolatilityModel, simulate

struct MAVolatilityModel{F <: Function}
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
    N = 2^Int(ceil(log2(m)))
    epsilon = randn( m )
    delta = rand( n )
    epsilon2 = Complex[epsilon; zeros(N - m)]
    a = Complex[[model.f(i) for i in 1:n]; zeros(N - n)]
    convolve!( epsilon2, a, a )
    x = exp.(cumsum( a[1:n] )) .* delta
    return (epsilon, x)
end
