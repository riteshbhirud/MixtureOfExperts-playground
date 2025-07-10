"""
    SoftMoEGating(k::Int, λ::Float32)

Soft routing where tokens are weighted combinations of experts.
From recent MoE research - allows gradient flow through all experts.
"""
struct SoftMoEGating <: GatingMechanism
    k::Int
    λ::Float32 
end

function compute_gates(gate::SoftMoEGating, router_logits::AbstractMatrix)
    scaled_logits = router_logits ./ gate.λ
    router_probs = softmax(scaled_logits; dims=1)
    
    num_experts, batch_size = size(router_logits)
    
    expert_indices = zeros(Int, gate.k, batch_size)
    expert_gates = zeros(Float32, num_experts, batch_size)
    
    for i in 1:batch_size
        topk_indices = partialsortperm(router_probs[:, i], 1:gate.k, rev=true)
        expert_indices[:, i] = topk_indices
    end
    
    expert_gates = router_probs
    
    return expert_indices, expert_gates, router_probs
end

"""
    HashGating(k::Int, hash_fn::Function)

Deterministic routing based on hash functions.
Useful for reproducibility and avoiding routing collapse.
"""
struct HashGating <: GatingMechanism
    k::Int
    num_experts::Int
    hash_fn::Function
end

HashGating(k::Int, num_experts::Int) = HashGating(k, num_experts, hash)

function compute_gates(gate::HashGating, router_logits::AbstractMatrix)
    num_experts, batch_size = size(router_logits)
    
    expert_indices = zeros(Int, gate.k, batch_size)
    expert_gates = ones(Float32, gate.k, batch_size) ./ gate.k
    
    for i in 1:batch_size
        token_hash = gate.hash_fn(i)
        
        selected = Int[]
        for j in 0:(gate.k-1)
            expert = (token_hash + j) % num_experts + 1
            push!(selected, expert)
        end
        
        expert_indices[:, i] = selected
    end
    
    router_probs = ones(Float32, num_experts, batch_size) ./ num_experts
    
    return expert_indices, expert_gates, router_probs
end

"""
    SharedExpertGating(num_shared::Int, base_gate::GatingMechanism)

DeepSeek/Qwen style with shared + routed experts.
Some experts are always active (shared), others are routed.
"""
struct SharedExpertGating <: GatingMechanism
    num_shared::Int
    base_gate::GatingMechanism
    num_experts::Int
end

function compute_gates(gate::SharedExpertGating, router_logits::AbstractMatrix)
    num_experts, batch_size = size(router_logits)
    
    routed_experts = (gate.num_shared + 1):gate.num_experts
    routed_logits = router_logits[routed_experts, :]
    
    routed_indices, routed_gates, routed_probs = compute_gates(gate.base_gate, routed_logits)
    
    routed_indices .+= gate.num_shared
    
    total_k = gate.num_shared + size(routed_indices, 1)
    expert_indices = zeros(Int, total_k, batch_size)
    expert_gates = zeros(Float32, total_k, batch_size)
    
    for i in 1:batch_size
        expert_indices[1:gate.num_shared, i] = 1:gate.num_shared
        expert_gates[1:gate.num_shared, i] .= 1.0f0 / total_k
        
        expert_indices[(gate.num_shared+1):end, i] = routed_indices[:, i]
        expert_gates[(gate.num_shared+1):end, i] = routed_gates[:, i] .* 
                                                    (1.0f0 - gate.num_shared / total_k)
    end
    
    router_probs = zeros(Float32, num_experts, batch_size)
    router_probs[1:gate.num_shared, :] .= 1.0f0 / num_experts
    router_probs[routed_experts, :] = routed_probs .* (1.0f0 - gate.num_shared / num_experts)
    
    return expert_indices, expert_gates, router_probs
end