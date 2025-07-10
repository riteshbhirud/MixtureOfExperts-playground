"""
    GatedExpert

Llama-style gated FFN expert: FFN(x) = w2(silu(w1(x)) * w3(x))
"""
struct GatedExpert{W1, W2, W3, A} <: Expert
    w1::W1 
    w2::W2  
    w3::W3  
    activation::A
end

Flux.@functor GatedExpert (w1, w2, w3)

function GatedExpert(input_dim::Int, hidden_dim::Int, output_dim::Int, activation)
    σ = sqrt(2.0f0 / input_dim)
    
    w1 = Dense(input_dim, hidden_dim, bias=false)
    w2 = Dense(hidden_dim, output_dim, bias=false)
    w3 = Dense(input_dim, hidden_dim, bias=false)
    
    w1.weight .= randn(Float32, size(w1.weight)) .* σ
    w2.weight .= randn(Float32, size(w2.weight)) .* σ
    w3.weight .= randn(Float32, size(w3.weight)) .* σ
    
    return GatedExpert(w1, w2, w3, activation)
end

function (expert::GatedExpert)(x; training::Bool = false)
    # Gated FFN: w2(act(w1(x)) * w3(x))
    gate = expert.activation.(expert.w1(x))
    up = expert.w3(x)
    return expert.w2(gate .* up)
end

silu(x) = x * sigmoid(x)