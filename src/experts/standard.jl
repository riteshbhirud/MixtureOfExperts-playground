"""
    Expert

Standard 2-layer FFN expert.
"""
abstract type Expert end

struct StandardExpert{W1, W2, B1, B2, A, D} <: Expert
    w1::W1
    w2::W2
    b1::B1
    b2::B2
    activation::A
    dropout::D
end

Flux.@functor StandardExpert (w1, w2, b1, b2)

function StandardExpert(input_dim::Int, hidden_dim::Int, output_dim::Int, 
                       activation; dropout::Float32 = 0.0f0, bias::Bool = true)
    σ1 = sqrt(2.0f0 / input_dim)
    σ2 = sqrt(2.0f0 / hidden_dim)
    
    w1 = Dense(input_dim, hidden_dim, bias=false)
    w2 = Dense(hidden_dim, output_dim, bias=false)
    
    w1.weight .= randn(Float32, size(w1.weight)) .* σ1
    w2.weight .= randn(Float32, size(w2.weight)) .* σ2
    
    b1 = bias ? zeros(Float32, hidden_dim) : nothing
    b2 = bias ? zeros(Float32, output_dim) : nothing
    
    drop = dropout > 0 ? Dropout(dropout) : nothing
    
    return StandardExpert(w1, w2, b1, b2, activation, drop)
end

function (expert::StandardExpert)(x; training::Bool = false)
    h = expert.w1(x)
    if !isnothing(expert.b1)
        h = h .+ expert.b1
    end
    h = expert.activation.(h)
    
    if !isnothing(expert.dropout) && training
        h = expert.dropout(h)
    end
    
    y = expert.w2(h)
    if !isnothing(expert.b2)
        y = y .+ expert.b2
    end
    
    return y
end