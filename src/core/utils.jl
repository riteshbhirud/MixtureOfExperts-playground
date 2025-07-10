# Utility functions

using LinearAlgebra
using Statistics

"""
    softplus(x)
    
Softplus activation: log(1 + exp(x))
"""
softplus(x) = log1p(exp(x))

"""
    entropy(p)
    
Compute entropy of probability distribution.
"""
entropy(p::AbstractArray) = -sum(p .* log.(p .+ 1e-8))

"""
    gumbel_softmax(logits, τ; training)
    
Gumbel-Softmax for differentiable discrete sampling.
"""
function gumbel_softmax(logits::AbstractMatrix, τ::Float32 = 1.0f0; training::Bool = false)
    if training
        U = rand(Float32, size(logits))
        G = -log.(-log.(U .+ 1e-8) .+ 1e-8)
        logits = (logits .+ G) ./ τ
    end
    
    return softmax(logits; dims=1)
end

"""
    load_balance_score(expert_counts)
    
Compute load balance score (1.0 is perfect balance).
"""
function load_balance_score(expert_counts::AbstractVector)
    n = length(expert_counts)
    total = sum(expert_counts)
    
    if total == 0
        return 1.0f0
    end
    
    ideal = total / n
    variance = sum((expert_counts .- ideal).^2) / n
    
    return 1.0f0 - sqrt(variance) / ideal
end

"""
    top_k_mask(logits, k)
    
Create mask for top-k selection.
"""
function top_k_mask(logits::AbstractMatrix, k::Int)
    mask = zeros(Bool, size(logits))
    
    for j in 1:size(logits, 2)
        topk_indices = partialsortperm(logits[:, j], 1:k, rev=true)
        mask[topk_indices, j] .= true
    end
    
    return mask
end