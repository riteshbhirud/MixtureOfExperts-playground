"""
    TopKGating(k::Int)

Stanford CS336 Top-K gating implementation.
Mathematical formulation from lecture:

h_t^l = Σ(i=1 to N) g_{i,t} FFN_i(u_t^l) + u_t^l

g_{i,t} = {
  s_{i,t}, if s_{i,t} ∈ TopK({s_{j,t} | 1 ≤ j ≤ N}, K)
  0, otherwise
}

s_{i,t} = Softmax_i(u_t^T e_i^l)
"""
struct TopKGating <: GatingMechanism
    k::Int
    use_softmax_renorm::Bool 
    use_raw_logits::Bool     
end

TopKGating(k::Int) = TopKGating(k, true, false)

function compute_gates(gate::TopKGating, router_logits::AbstractMatrix)

    # s_{i,t} = Softmax_i(u_t^T e_i^l)
    router_probs = softmax(router_logits; dims=1)
    
    num_experts, batch_size = size(router_logits)
    expert_indices = zeros(Int, gate.k, batch_size)
    expert_gates = zeros(Float32, gate.k, batch_size)
    
    for t in 1:batch_size  # t = token index
        # Find TopK: s_{i,t} ∈ TopK({s_{j,t} | 1 ≤ j ≤ N}, K)
        topk_indices = partialsortperm(router_probs[:, t], 1:gate.k, rev=true)
        expert_indices[:, t] = topk_indices
        
        # g_{i,t} = s_{i,t} if in TopK, 0 otherwise
        selected_probs = router_probs[topk_indices, t]
        expert_gates[:, t] = selected_probs ./ sum(selected_probs)  
    end
    
    return expert_indices, expert_gates, router_probs
end

"""
    StochasticTopKGating(k::Int, noise_std::Function)

Gaussian perturbation from Stanford CS336 / Shazeer et al 2017:
G(x) = Softmax(KeepTopK(H(x), k))
H(x)_i = (x · W_g)_i + StandardNormal() · Softplus((x · W_noise)_i)
"""
struct StochasticTopKGating <: GatingMechanism
    k::Int
    train_only::Bool
end

function compute_gates(gate::StochasticTopKGating, router_logits::AbstractMatrix, 
                      noise_logits::Union{Nothing, AbstractMatrix} = nothing;
                      training::Bool = false)
    if training || !gate.train_only
        if !isnothing(noise_logits)
            noise = randn(Float32, size(router_logits)) .* softplus.(noise_logits)
            router_logits = router_logits .+ noise
        end
    end
    
    return compute_gates(TopKGating(gate.k), router_logits)
end