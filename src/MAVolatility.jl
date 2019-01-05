
struct MAVolatilityModel{T <: Real}
    a::Vector{T}
end

function convolve!( u::AbstractVector{Complex{T}},
                   v::AbstractVector{Complex{T}},
                   w::AbstractVector{Complex{T}} ) where {T <: Number}
    fft!( u )
    fft!( v )
    w[1:32] = u .* v
    ifft!( w )
end

function simulate( model::MAVolatilityModel{T}, n::UInt ) where {T}
    m = 2*n-1
    N = 2^Int(ceil(log(m)))
    epsilon = randn( m )
    delta = rand( n )
end
